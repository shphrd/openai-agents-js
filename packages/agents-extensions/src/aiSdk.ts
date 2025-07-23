import type {
  JSONSchema7,
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2FunctionTool,
  LanguageModelV2Message,
  LanguageModelV2Prompt,
  LanguageModelV2ProviderDefinedTool,
  LanguageModelV2ToolCallPart,
  LanguageModelV2ToolResultPart,
} from '@ai-sdk/provider';
import {
  createGenerationSpan,
  Model,
  ModelRequest,
  ModelResponse,
  protocol,
  resetCurrentSpan,
  ResponseStreamEvent,
  SerializedHandoff,
  SerializedOutputType,
  SerializedTool,
  setCurrentSpan,
  Usage,
  UserError,
  withGenerationSpan,
  getLogger,
} from '@openai/agents';
import { isZodObject } from '@openai/agents/utils';

/**
 * @internal
 * Converts a list of model items to a list of language model v1 messages.
 *
 * @param model - The model to use.
 * @param items - The items to convert.
 * @returns The list of language model v1 messages.
 */
export function itemsToLanguageV2Messages(
  model: LanguageModelV2,
  items: protocol.ModelItem[],
): LanguageModelV2Message[] {
  const messages: LanguageModelV2Message[] = [];
  let currentAssistantMessage: LanguageModelV2Message | undefined;

  for (const item of items) {
    if (item.type === 'message' || typeof item.type === 'undefined') {
      const { role, content, providerData } = item;
      if (role === 'system') {
        messages.push({
          role: 'system',
          content: content,
          providerOptions: {
            ...(providerData ?? {}),
          },
        });
        continue;
      }

      if (role === 'user') {
        messages.push({
          role,
          content:
            typeof content === 'string'
              ? [{ type: 'text', text: content }]
              : content.map((c) => {
                  const { providerData: contentProviderData } = c;
                  if (c.type === 'input_text') {
                    return {
                      type: 'text',
                      text: c.text,
                      providerOptions: {
                        ...(contentProviderData ?? {}),
                      },
                    };
                  }
                  if (c.type === 'input_image') {
                    const url = new URL(c.image);
                    return {
                      type: 'file',
                      data: url.toString(),
                      mediaType: 'image/*',
                      providerOptions: {
                        ...(contentProviderData ?? {}),
                      },
                    };
                  }
                  if (c.type === 'input_file') {
                    if (typeof c.file !== 'string') {
                      throw new UserError('File ID is not supported');
                    }
                    return {
                      type: 'file',
                      data: c.file,
                      mediaType: 'application/octet-stream',
                      providerOptions: {
                        ...(contentProviderData ?? {}),
                      },
                    };
                  }
                  throw new UserError(`Unknown content type: ${c.type}`);
                }),
          providerOptions: {
            ...(providerData ?? {}),
          },
        });
        continue;
      }

      if (role === 'assistant') {
        if (currentAssistantMessage) {
          messages.push(currentAssistantMessage);
          currentAssistantMessage = undefined;
        }

        messages.push({
          role,
          content: content
            .filter((c) => c.type === 'input_text' || c.type === 'output_text')
            .map((c) => {
              const { providerData: contentProviderData } = c;
              if (c.type === 'output_text') {
                return {
                  type: 'text',
                  text: c.text,
                  providerOptions: {
                    ...(contentProviderData ?? {}),
                  },
                };
              }
              if (c.type === 'input_text') {
                return {
                  type: 'text',
                  text: c.text,
                  providerOptions: {
                    ...(contentProviderData ?? {}),
                  },
                };
              }
              const exhaustiveCheck = c satisfies never;
              throw new UserError(`Unknown content type: ${exhaustiveCheck}`);
            }),
          providerOptions: {
            ...(providerData ?? {}),
          },
        });
        continue;
      }

      const exhaustiveMessageTypeCheck = item satisfies never;
      throw new Error(`Unknown message type: ${exhaustiveMessageTypeCheck}`);
    } else if (item.type === 'function_call') {
      if (!currentAssistantMessage) {
        currentAssistantMessage = {
          role: 'assistant',
          content: [],
          providerOptions: {
            ...(item.providerData ?? {}),
          },
        };
      }

      if (
        Array.isArray(currentAssistantMessage.content) &&
        currentAssistantMessage.role === 'assistant'
      ) {
        const content: LanguageModelV2ToolCallPart = {
          type: 'tool-call',
          toolCallId: item.callId,
          toolName: item.name,
          input: parseArguments(item.arguments),
          providerOptions: {
            ...(item.providerData ?? {}),
          },
        };
        currentAssistantMessage.content.push(content);
      }
      continue;
    } else if (item.type === 'function_call_result') {
      if (currentAssistantMessage) {
        messages.push(currentAssistantMessage);
        currentAssistantMessage = undefined;
      }
      const toolResult: LanguageModelV2ToolResultPart = {
        type: 'tool-result',
        toolCallId: item.callId,
        toolName: item.name,
        output:
          typeof item.output === 'string'
            ? { type: 'text', value: item.output }
            : Array.isArray(item.output)
              ? {
                  type: 'text',
                  value: item.output
                    .map((o) =>
                      o.type === 'text'
                        ? o.text
                        : o.type === 'image'
                          ? `[Image: ${o.mediaType}]`
                          : String(o),
                    )
                    .join('\n'),
                }
              : { type: 'text', value: String(item.output) },
        providerOptions: {
          ...(item.providerData ?? {}),
        },
      };
      messages.push({
        role: 'tool',
        content: [toolResult],
        providerOptions: {
          ...(item.providerData ?? {}),
        },
      });
      continue;
    }

    if (item.type === 'hosted_tool_call') {
      throw new UserError('Hosted tool calls are not supported');
    }

    if (item.type === 'computer_call') {
      throw new UserError('Computer calls are not supported');
    }

    if (item.type === 'computer_call_result') {
      throw new UserError('Computer call results are not supported');
    }

    if (
      item.type === 'reasoning' &&
      item.content.length > 0 &&
      typeof item.content[0].text === 'string'
    ) {
      messages.push({
        role: 'assistant',
        content: [
          {
            type: 'reasoning',
            text: item.content[0].text,
            providerOptions: { ...(item.providerData ?? {}) },
          },
        ],
        providerOptions: {
          ...(item.providerData ?? {}),
        },
      });
      continue;
    }

    if (item.type === 'unknown') {
      messages.push({ ...(item.providerData ?? {}) } as LanguageModelV2Message);
      continue;
    }

    if (item) {
      throw new UserError(`Unknown item type: ${item.type}`);
    }

    const itemType = item satisfies never;
    throw new UserError(`Unknown item type: ${itemType}`);
  }

  if (currentAssistantMessage) {
    messages.push(currentAssistantMessage);
  }

  return messages;
}

/**
 * @internal
 * Converts a handoff to a language model v1 tool.
 *
 * @param model - The model to use.
 * @param handoff - The handoff to convert.
 */
function handoffToLanguageV2Tool(
  model: LanguageModelV2,
  handoff: SerializedHandoff,
): LanguageModelV2FunctionTool {
  return {
    type: 'function',
    name: handoff.toolName,
    description: handoff.toolDescription,
    inputSchema: handoff.inputJsonSchema as JSONSchema7,
  };
}

/**
 * @internal
 * Converts a tool to a language model v1 tool.
 *
 * @param model - The model to use.
 * @param tool - The tool to convert.
 */
export function toolToLanguageV2Tool(
  model: LanguageModelV2,
  tool: SerializedTool,
): LanguageModelV2FunctionTool | LanguageModelV2ProviderDefinedTool {
  if (tool.type === 'function') {
    return {
      type: 'function',
      name: tool.name,
      description: tool.description,
      inputSchema: tool.parameters as JSONSchema7,
    };
  }

  if (tool.type === 'hosted_tool') {
    return {
      type: 'provider-defined',
      id: `${model.provider}.${tool.name}`,
      name: tool.name,
      args: tool.providerData?.args ?? {},
    };
  }

  if (tool.type === 'computer') {
    return {
      type: 'provider-defined',
      id: `${model.provider}.${tool.name}`,
      name: tool.name,
      args: {
        environment: tool.environment,
        display_width: tool.dimensions[0],
        display_height: tool.dimensions[1],
      },
    };
  }

  const exhaustiveCheck: never = tool;
  throw new Error(`Unsupported tool type: ${exhaustiveCheck}`);
}

/**
 * @internal
 * Converts an output type to a language model v1 response format.
 *
 * @param outputType - The output type to convert.
 * @returns The language model v1 response format.
 */
export function getResponseFormat(
  outputType: SerializedOutputType,
): LanguageModelV2CallOptions['responseFormat'] {
  if (outputType === 'text') {
    return {
      type: 'text',
    };
  }

  return {
    type: 'json',
    name: outputType.name,
    schema: outputType.schema,
  };
}

/**
 * Wraps a model from the AI SDK that adheres to the LanguageModelV2 spec to be used used as a model
 * in the OpenAI Agents SDK to use other models.
 *
 * While you can use this with the OpenAI models, it is recommended to use the default OpenAI model
 * provider instead.
 *
 * If tracing is enabled, the model will send generation spans to your traces processor.
 *
 * ```ts
 * import { aisdk } from '@openai/agents-extensions';
 * import { openai } from '@ai-sdk/openai';
 *
 * const model = aisdk(openai('gpt-4o'));
 *
 * const agent = new Agent({
 *   name: 'My Agent',
 *   model
 * });
 * ```
 *
 * @param model - The Vercel AI SDK model to wrap.
 * @returns The wrapped model.
 */
export class AiSdkModel implements Model {
  #model: LanguageModelV2;
  #logger = getLogger('openai-agents:extensions:ai-sdk');
  constructor(model: LanguageModelV2) {
    this.#model = model;
  }

  async getResponse(request: ModelRequest) {
    return withGenerationSpan(async (span) => {
      try {
        span.spanData.model = this.#model.provider + ':' + this.#model.modelId;
        span.spanData.model_config = {
          provider: this.#model.provider,
          model_impl: 'ai-sdk',
        };

        let input: LanguageModelV2Prompt =
          typeof request.input === 'string'
            ? [
                {
                  role: 'user',
                  content: [{ type: 'text', text: request.input }],
                },
              ]
            : itemsToLanguageV2Messages(this.#model, request.input);

        if (request.systemInstructions) {
          input = [
            {
              role: 'system',
              content: request.systemInstructions,
            },
            ...input,
          ];
        }

        const tools = request.tools.map((tool) =>
          toolToLanguageV2Tool(this.#model, tool),
        );

        request.handoffs.forEach((handoff) => {
          tools.push(handoffToLanguageV2Tool(this.#model, handoff));
        });

        if (span && request.tracing === true) {
          span.spanData.input = input;
        }

        if (isZodObject(request.outputType)) {
          throw new UserError('Zod output type is not yet supported');
        }

        const responseFormat: LanguageModelV2CallOptions['responseFormat'] =
          getResponseFormat(request.outputType);

        const aiSdkRequest: LanguageModelV2CallOptions = {
          tools,
          prompt: input,
          temperature: request.modelSettings.temperature,
          topP: request.modelSettings.topP,
          frequencyPenalty: request.modelSettings.frequencyPenalty,
          presencePenalty: request.modelSettings.presencePenalty,
          maxOutputTokens: request.modelSettings.maxTokens,
          responseFormat,
          abortSignal: request.signal,

          ...(request.modelSettings.providerData ?? {}),
        };

        if (this.#logger.dontLogModelData) {
          this.#logger.debug('Request sent');
        } else {
          this.#logger.debug('Request:', aiSdkRequest);
        }

        const result = await this.#model.doGenerate(aiSdkRequest);

        const output: ModelResponse['output'] = [];

        // Parse v2 content array to extract tool calls and text
        const toolCalls = result.content.filter(
          (item) => item.type === 'tool-call',
        );
        const textContent = result.content.filter(
          (item) => item.type === 'text',
        );

        toolCalls.forEach((toolCall) => {
          output.push({
            type: 'function_call',
            callId: toolCall.toolCallId,
            name: toolCall.toolName,
            arguments:
              typeof toolCall.input === 'string'
                ? toolCall.input
                : JSON.stringify(toolCall.input),
            status: 'completed',
            providerData:
              textContent.length === 0 ? result.providerMetadata : undefined,
          });
        });

        // Some of other platforms may return both tool calls and text.
        // Putting a text message here will let the agent loop to complete,
        // so adding this item only when the tool calls are empty.
        // Note that the same support is not available for streaming mode.
        if (toolCalls.length === 0 && textContent.length > 0) {
          const combinedText = textContent.map((t) => t.text).join('');
          output.push({
            type: 'message',
            content: [{ type: 'output_text', text: combinedText }],
            role: 'assistant',
            status: 'completed',
            providerData: result.providerMetadata,
          });
        }

        if (span && request.tracing === true) {
          span.spanData.output = output;
        }

        const response = {
          responseId: result.response?.id ?? 'FAKE_ID',
          usage: new Usage({
            inputTokens: Number.isNaN(result.usage?.inputTokens)
              ? 0
              : (result.usage?.inputTokens ?? 0),
            outputTokens: Number.isNaN(result.usage?.outputTokens)
              ? 0
              : (result.usage?.outputTokens ?? 0),
            totalTokens:
              (Number.isNaN(result.usage?.inputTokens)
                ? 0
                : (result.usage?.inputTokens ?? 0)) +
              (Number.isNaN(result.usage?.outputTokens)
                ? 0
                : (result.usage?.outputTokens ?? 0)),
          }),
          output,
          providerData: result,
        } as const;

        if (span && request.tracing === true) {
          span.spanData.usage = {
            input_tokens: response.usage.inputTokens,
            output_tokens: response.usage.outputTokens,
            total_tokens: response.usage.totalTokens,
          };
        }

        if (this.#logger.dontLogModelData) {
          this.#logger.debug('Response ready');
        } else {
          this.#logger.debug('Response:', response);
        }

        return response;
      } catch (error) {
        if (error instanceof Error) {
          span.setError({
            message: request.tracing === true ? error.message : 'Unknown error',
            data: {
              error:
                request.tracing === true
                  ? String(error)
                  : error instanceof Error
                    ? error.name
                    : undefined,
            },
          });
        } else {
          span.setError({
            message: 'Unknown error',
            data: {
              error:
                request.tracing === true
                  ? String(error)
                  : error instanceof Error
                    ? error.name
                    : undefined,
            },
          });
        }
        throw error;
      }
    });
  }

  async *getStreamedResponse(
    request: ModelRequest,
  ): AsyncIterable<ResponseStreamEvent> {
    const span = request.tracing ? createGenerationSpan() : undefined;
    try {
      if (span) {
        span.start();
        setCurrentSpan(span);
      }

      if (span?.spanData) {
        span.spanData.model = this.#model.provider + ':' + this.#model.modelId;
        span.spanData.model_config = {
          provider: this.#model.provider,
          model_impl: 'ai-sdk',
        };
      }

      let input: LanguageModelV2Prompt =
        typeof request.input === 'string'
          ? [
              {
                role: 'user',
                content: [{ type: 'text', text: request.input }],
              },
            ]
          : itemsToLanguageV2Messages(this.#model, request.input);

      if (request.systemInstructions) {
        input = [
          {
            role: 'system',
            content: request.systemInstructions,
          },
          ...input,
        ];
      }

      const tools = request.tools.map((tool) =>
        toolToLanguageV2Tool(this.#model, tool),
      );

      request.handoffs.forEach((handoff) => {
        tools.push(handoffToLanguageV2Tool(this.#model, handoff));
      });

      if (span && request.tracing === true) {
        span.spanData.input = input;
      }

      const responseFormat: LanguageModelV2CallOptions['responseFormat'] =
        getResponseFormat(request.outputType);

      const aiSdkRequest: LanguageModelV2CallOptions = {
        tools,
        prompt: input,
        temperature: request.modelSettings.temperature,
        topP: request.modelSettings.topP,
        frequencyPenalty: request.modelSettings.frequencyPenalty,
        presencePenalty: request.modelSettings.presencePenalty,
        maxOutputTokens: request.modelSettings.maxTokens,
        responseFormat,
        abortSignal: request.signal,
        ...(request.modelSettings.providerData ?? {}),
      };

      if (this.#logger.dontLogModelData) {
        this.#logger.debug('Request received (streamed)');
      } else {
        this.#logger.debug('Request (streamed):', aiSdkRequest);
      }

      const { stream } = await this.#model.doStream(aiSdkRequest);

      let started = false;
      let responseId: string | undefined;
      let usagePromptTokens = 0;
      let usageCompletionTokens = 0;
      const functionCalls: Record<string, protocol.FunctionCallItem> = {};
      let textOutput: protocol.OutputText | undefined;

      for await (const part of stream) {
        if (!started) {
          started = true;
          yield { type: 'response_started' };
        }

        yield { type: 'model', event: part };

        switch (part.type) {
          case 'text-delta': {
            if (!textOutput) {
              textOutput = { type: 'output_text', text: '' };
            }
            textOutput.text += part.delta;
            yield { type: 'output_text_delta', delta: part.delta };
            break;
          }
          case 'tool-input-start': {
            functionCalls[part.id] = {
              type: 'function_call',
              callId: part.id,
              name: part.toolName,
              arguments: '',
              status: 'completed',
            };
            break;
          }
          case 'tool-input-delta': {
            const fc = functionCalls[part.id];
            if (fc) {
              fc.arguments += part.delta;
            }
            break;
          }
          case 'tool-input-end': {
            // Tool input streaming complete - arguments are finalized
            const fc = functionCalls[part.id];
            if (fc) {
              // Arguments are already accumulated as JSON string, which is what agents framework expects
              // No additional processing needed
            }
            break;
          }
          case 'stream-start': {
            // Stream initialization - no action needed
            break;
          }
          case 'response-metadata': {
            if (part.id) {
              responseId = part.id;
            }
            break;
          }
          case 'finish': {
            usagePromptTokens = Number.isNaN(part.usage?.inputTokens)
              ? 0
              : (part.usage?.inputTokens ?? 0);
            usageCompletionTokens = Number.isNaN(part.usage?.outputTokens)
              ? 0
              : (part.usage?.outputTokens ?? 0);
            break;
          }
          case 'error': {
            throw part.error;
          }
          default:
            break;
        }
      }

      const outputs: protocol.OutputModelItem[] = [];
      if (textOutput) {
        outputs.push({
          type: 'message',
          role: 'assistant',
          content: [textOutput],
          status: 'completed',
        });
      }
      for (const fc of Object.values(functionCalls)) {
        outputs.push(fc);
      }

      const finalEvent: protocol.StreamEventResponseCompleted = {
        type: 'response_done',
        response: {
          id: responseId ?? 'FAKE_ID',
          usage: {
            inputTokens: usagePromptTokens,
            outputTokens: usageCompletionTokens,
            totalTokens: usagePromptTokens + usageCompletionTokens,
          },
          output: outputs,
        },
      };

      if (span && request.tracing === true) {
        span.spanData.output = outputs;
        span.spanData.usage = {
          input_tokens: finalEvent.response.usage.inputTokens,
          output_tokens: finalEvent.response.usage.outputTokens,
          total_tokens: finalEvent.response.usage.totalTokens,
        };
      }

      if (this.#logger.dontLogModelData) {
        this.#logger.debug('Response ready (streamed)');
      } else {
        this.#logger.debug('Response (streamed):', finalEvent.response);
      }

      yield finalEvent;
    } catch (error) {
      if (span) {
        span.setError({
          message: 'Error streaming response',
          data: {
            error:
              request.tracing === true
                ? String(error)
                : error instanceof Error
                  ? error.name
                  : undefined,
          },
        });
      }
      throw error;
    } finally {
      if (span) {
        span.end();
        resetCurrentSpan();
      }
    }
  }
}

/**
 * Wraps a model from the AI SDK that adheres to the LanguageModelV2 spec to be used used as a model
 * in the OpenAI Agents SDK to use other models.
 *
 * While you can use this with the OpenAI models, it is recommended to use the default OpenAI model
 * provider instead.
 *
 * If tracing is enabled, the model will send generation spans to your traces processor.
 *
 * ```ts
 * import { aisdk } from '@openai/agents-extensions';
 * import { openai } from '@ai-sdk/openai';
 *
 * const model = aisdk(openai('gpt-4o'));
 *
 * const agent = new Agent({
 *   name: 'My Agent',
 *   model
 * });
 * ```
 *
 * @param model - The Vercel AI SDK model to wrap.
 * @returns The wrapped model.
 */
export function aisdk(model: LanguageModelV2) {
  return new AiSdkModel(model);
}

export function parseArguments(args: string | undefined | null): any {
  if (!args) {
    return {};
  }

  try {
    return JSON.parse(args);
  } catch (_) {
    return {};
  }
}
