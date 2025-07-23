import { describe, test, expect, vi } from 'vitest';
import {
  AiSdkModel,
  getResponseFormat,
  itemsToLanguageV2Messages,
  parseArguments,
  toolToLanguageV2Tool,
} from '../src/aiSdk';
import { protocol, withTrace, UserError } from '@openai/agents';
import { ReadableStream } from 'node:stream/web';
import type { LanguageModelV2 } from '@ai-sdk/provider';
import type { SerializedOutputType } from '@openai/agents';
import { MockLanguageModelV2, simulateReadableStream } from 'ai/test';

function createMockModel(
  config: {
    doGenerate?:
      | LanguageModelV2['doGenerate']
      | Awaited<ReturnType<LanguageModelV2['doGenerate']>>;
    doStream?:
      | LanguageModelV2['doStream']
      | Awaited<ReturnType<LanguageModelV2['doStream']>>;
  } = {},
): MockLanguageModelV2 {
  return new MockLanguageModelV2({
    provider: 'test',
    modelId: 'test-model',
    doGenerate: config.doGenerate || {
      content: [{ type: 'text', text: '' }],
      finishReason: 'stop',
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      rawCall: { rawPrompt: '', rawSettings: {} },
      warnings: [],
    },
    doStream: config.doStream || {
      stream: new ReadableStream(),
      rawCall: { rawPrompt: '', rawSettings: {} },
    },
  });
}

// Use AI SDK v5 simulateReadableStream for better streaming tests
function createStreamingMock(parts: any[]): MockLanguageModelV2 {
  return createMockModel({
    doStream: {
      stream: simulateReadableStream({
        chunks: parts,
        chunkDelayInMs: 0, // No delay for tests
      }),
      rawCall: { rawPrompt: '', rawSettings: {} },
    },
  });
}

describe('getResponseFormat', () => {
  test('converts text output type', () => {
    const outputType: SerializedOutputType = 'text';
    const result = getResponseFormat(outputType);
    expect(result).toEqual({ type: 'text' });
  });

  test('converts json schema output type', () => {
    const outputType: SerializedOutputType = {
      type: 'json_schema',
      name: 'output',
      strict: false,
      schema: {
        type: 'object',
        properties: {
          name: { type: 'string' },
        },
        required: ['name'],
        additionalProperties: false,
      },
    };
    const result = getResponseFormat(outputType);
    expect(result).toEqual({
      type: 'json',
      name: outputType.name,
      schema: outputType.schema,
    });
  });
});

describe('itemsToLanguageV2Messages', () => {
  test('converts user text and function call items', () => {
    const items: protocol.ModelItem[] = [
      {
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: 'hi',
            providerData: { test: { cacheControl: { type: 'ephemeral' } } },
          },
        ],
      } as any,
      {
        type: 'function_call',
        callId: '1',
        name: 'foo',
        arguments: '{}',
        providerData: { a: 1 },
      } as any,
      {
        type: 'function_call_result',
        callId: '1',
        name: 'foo',
        output: 'out',
        providerData: { b: 2 },
      } as any,
    ];

    const msgs = itemsToLanguageV2Messages(createMockModel(), items);
    expect(msgs).toEqual([
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: 'hi',
            providerOptions: { test: { cacheControl: { type: 'ephemeral' } } },
          },
        ],
        providerOptions: {},
      },
      {
        role: 'assistant',
        content: [
          {
            type: 'tool-call',
            toolCallId: '1',
            toolName: 'foo',
            input: {},
            providerOptions: { a: 1 },
          },
        ],
        providerOptions: { a: 1 },
      },
      {
        role: 'tool',
        content: [
          {
            type: 'tool-result',
            toolCallId: '1',
            toolName: 'foo',
            output: { type: 'text', value: 'out' },
            providerOptions: { b: 2 },
          },
        ],
        providerOptions: { b: 2 },
      },
    ]);
  });

  test('throws on built-in tool calls', () => {
    const items: protocol.ModelItem[] = [
      { type: 'hosted_tool_call', name: 'search' } as any,
    ];
    expect(() => itemsToLanguageV2Messages(createMockModel(), items)).toThrow();
  });

  test('converts user images, function results and reasoning items', () => {
    const items: protocol.ModelItem[] = [
      {
        role: 'user',
        content: [
          { type: 'input_text', text: 'hi' },
          { type: 'input_image', image: 'http://x/img' },
        ],
      } as any,
      {
        type: 'function_call',
        callId: '1',
        name: 'do',
        arguments: '{}',
      } as any,
      {
        type: 'function_call_result',
        callId: '1',
        name: 'do',
        output: 'out',
      } as any,
      { type: 'reasoning', content: [{ text: 'why' }] } as any,
    ];
    const msgs = itemsToLanguageV2Messages(createMockModel(), items);
    expect(msgs).toEqual([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'hi', providerOptions: {} },
          {
            type: 'file',
            data: 'http://x/img',
            mediaType: 'image/*',
            providerOptions: {},
          },
        ],
        providerOptions: {},
      },
      {
        role: 'assistant',
        content: [
          {
            type: 'tool-call',
            toolCallId: '1',
            toolName: 'do',
            input: {},
            providerOptions: {},
          },
        ],
        providerOptions: {},
      },
      {
        role: 'tool',
        content: [
          {
            type: 'tool-result',
            toolCallId: '1',
            toolName: 'do',
            output: { type: 'text', value: 'out' },
            providerOptions: {},
          },
        ],
        providerOptions: {},
      },
      {
        role: 'assistant',
        content: [{ type: 'reasoning', text: 'why', providerOptions: {} }],
        providerOptions: {},
      },
    ]);
  });

  test('handles undefined providerData without throwing', () => {
    const items: protocol.ModelItem[] = [
      {
        role: 'user',
        content: [{ type: 'input_text', text: 'hi' }],
        providerData: undefined,
      } as any,
    ];
    expect(() =>
      itemsToLanguageV2Messages(createMockModel(), items),
    ).not.toThrow();
    const msgs = itemsToLanguageV2Messages(createMockModel(), items);
    expect(msgs).toEqual([
      {
        role: 'user',
        content: [{ type: 'text', text: 'hi', providerOptions: {} }],
        providerOptions: {},
      },
    ]);
  });

  test('throws UserError for unsupported content or unknown item type', () => {
    const bad: protocol.ModelItem[] = [
      { role: 'user', content: [{ type: 'bad' as any }] } as any,
    ];
    expect(() => itemsToLanguageV2Messages(createMockModel(), bad)).toThrow(
      UserError,
    );

    const unknown: protocol.ModelItem[] = [{ type: 'bogus' } as any];
    expect(() => itemsToLanguageV2Messages(createMockModel(), unknown)).toThrow(
      UserError,
    );
  });
});

describe('toolToLanguageV2Tool', () => {
  const model = createMockModel();
  test('maps function tools', () => {
    const tool = {
      type: 'function',
      name: 'foo',
      description: 'd',
      parameters: {} as any,
    } as any;
    expect(toolToLanguageV2Tool(model, tool)).toEqual({
      type: 'function',
      name: 'foo',
      description: 'd',
      inputSchema: {},
    });
  });

  test('maps builtin tools', () => {
    const tool = {
      type: 'hosted_tool',
      name: 'search',
      providerData: { args: { q: 1 } },
    } as any;
    expect(toolToLanguageV2Tool(model, tool)).toEqual({
      type: 'provider-defined',
      id: `${model.provider}.search`,
      name: 'search',
      args: { q: 1 },
    });
  });

  test('maps computer tools', () => {
    const tool = {
      type: 'computer',
      name: 'comp',
      environment: 'env',
      dimensions: [2, 3],
    } as any;
    expect(toolToLanguageV2Tool(model, tool)).toEqual({
      type: 'provider-defined',
      id: `${model.provider}.comp`,
      name: 'comp',
      args: { environment: 'env', display_width: 2, display_height: 3 },
    });
  });

  test('throws on unknown type', () => {
    const tool = { type: 'x', name: 'u' } as any;
    expect(() => toolToLanguageV2Tool(model, tool)).toThrow();
  });
});

describe('AiSdkModel.getResponse', () => {
  test('handles text output', async () => {
    const model = new AiSdkModel(
      createMockModel({
        doGenerate: {
          content: [{ type: 'text', text: 'ok' }],
          finishReason: 'stop',
          usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
          providerMetadata: { p: 1 },
          rawCall: { rawPrompt: '', rawSettings: {} },
          warnings: [],
        },
      }),
    );

    const res = await withTrace('t', () =>
      model.getResponse({
        input: 'hi',
        tools: [],
        handoffs: [],
        modelSettings: {},
        outputType: 'text',
        tracing: false,
      } as any),
    );

    expect(res.output).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: 'ok' }],
        status: 'completed',
        providerData: { p: 1 },
      },
    ]);
  });

  test('aborts when signal already aborted', async () => {
    const abort = new AbortController();
    abort.abort();
    const doGenerate = vi.fn(async (opts: any) => {
      if (opts.abortSignal?.aborted) {
        throw new Error('aborted');
      }
      return {
        content: [{ type: 'text', text: 'should not' }],
        finishReason: 'stop',
        usage: { inputTokens: 0, outputTokens: 0 },
        rawCall: { rawPrompt: '', rawSettings: {} },
      };
    });
    const model = new AiSdkModel(
      createMockModel({
        doGenerate,
      }),
    );

    await expect(
      withTrace('t', () =>
        model.getResponse({
          input: 'hi',
          tools: [],
          handoffs: [],
          modelSettings: {},
          outputType: 'text',
          tracing: false,
          signal: abort.signal,
        } as any),
      ),
    ).rejects.toThrow('aborted');
    expect(doGenerate).toHaveBeenCalled();
  });

  test('handles function call output', async () => {
    const model = new AiSdkModel(
      createMockModel({
        doGenerate: {
          content: [
            {
              type: 'tool-call',
              toolCallId: 'c1',
              toolName: 'foo',
              input: {} as any,
            },
          ],
          finishReason: 'stop',
          usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
          providerMetadata: { p: 1 },
          rawCall: { rawPrompt: '', rawSettings: {} },
          warnings: [],
        },
      }),
    );

    const res = await withTrace('t', () =>
      model.getResponse({
        input: 'hi',
        tools: [],
        handoffs: [],
        modelSettings: {},
        outputType: 'text',
        tracing: false,
      } as any),
    );

    expect(res.output).toEqual([
      {
        type: 'function_call',
        callId: 'c1',
        name: 'foo',
        arguments: {},
        status: 'completed',
        providerData: { p: 1 },
      },
    ]);
  });

  test('propagates errors', async () => {
    const model = new AiSdkModel(
      createMockModel({
        doGenerate: async () => {
          throw new Error('bad');
        },
      }),
    );

    await expect(
      withTrace('t', () =>
        model.getResponse({
          input: 'hi',
          tools: [],
          handoffs: [],
          modelSettings: {},
          outputType: 'text',
          tracing: false,
        } as any),
      ),
    ).rejects.toThrow('bad');
  });

  test('prepends system instructions to prompt for doGenerate', async () => {
    let received: any;
    const model = new AiSdkModel(
      createMockModel({
        doGenerate: async (options) => {
          received = options.prompt;
          return {
            content: [{ type: 'text', text: '' }],
            finishReason: 'stop',
            usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
            providerMetadata: {},
            rawCall: { rawPrompt: '', rawSettings: {} },
            warnings: [],
          };
        },
      }),
    );

    await withTrace('t', () =>
      model.getResponse({
        systemInstructions: 'inst',
        input: 'hi',
        tools: [],
        handoffs: [],
        modelSettings: {},
        outputType: 'text',
        tracing: false,
      } as any),
    );

    expect(received[0]).toEqual({
      role: 'system',
      content: 'inst',
    });
  });

  test('handles NaN usage in doGenerate', async () => {
    const model = new AiSdkModel(
      createMockModel({
        doGenerate: {
          content: [{ type: 'text', text: '' }],
          finishReason: 'stop',
          usage: {
            inputTokens: Number.NaN,
            outputTokens: Number.NaN,
            totalTokens: Number.NaN,
          },
          providerMetadata: {},
          rawCall: { rawPrompt: '', rawSettings: {} },
          warnings: [],
        },
      }),
    );

    const res = await withTrace('t', () =>
      model.getResponse({
        input: 'hi',
        tools: [],
        handoffs: [],
        modelSettings: {},
        outputType: 'text',
        tracing: false,
      } as any),
    );

    expect(res.usage).toEqual({
      requests: 1,
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      inputTokensDetails: [],
      outputTokensDetails: [],
    });
  });
});

describe('AiSdkModel.getStreamedResponse', () => {
  test('streams events and completes', async () => {
    const parts = [
      { type: 'text-delta', delta: 'a' },
      {
        type: 'tool-input-start',
        id: 'c1',
        toolName: 'foo',
      },
      {
        type: 'tool-input-delta',
        id: 'c1',
        delta: '{"k":"v"}',
      },
      { type: 'response-metadata', id: 'id1' },
      {
        type: 'finish',
        finishReason: 'stop',
        usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
      },
    ];
    const model = new AiSdkModel(createStreamingMock(parts));

    const events: any[] = [];
    for await (const ev of model.getStreamedResponse({
      input: 'hi',
      tools: [],
      handoffs: [],
      modelSettings: {},
      outputType: 'text',
      tracing: false,
    } as any)) {
      events.push(ev);
    }

    const final = events.at(-1);
    expect(final.type).toBe('response_done');
    expect(final.response.output).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: 'a' }],
        status: 'completed',
      },
      {
        type: 'function_call',
        callId: 'c1',
        name: 'foo',
        arguments: '{"k":"v"}',
        status: 'completed',
      },
    ]);
  });

  test('propagates stream errors', async () => {
    const err = new Error('bad');
    const parts = [{ type: 'error', error: err }];
    const model = new AiSdkModel(createStreamingMock(parts));

    await expect(async () => {
      const iter = model.getStreamedResponse({
        input: 'hi',
        tools: [],
        handoffs: [],
        modelSettings: {},
        outputType: 'text',
        tracing: false,
      } as any);

      for await (const ev of iter) {
        if (ev.type === 'response_done') {
          expect(ev.response.id).toBeDefined();
        } else if (ev.type === 'model') {
          expect(ev.event).toBeDefined();
        }
      }
    }).rejects.toThrow('bad');
  });

  test('aborts streaming when signal already aborted', async () => {
    const abort = new AbortController();
    abort.abort();
    const doStream = vi.fn(async (opts: any) => {
      if (opts.abortSignal?.aborted) {
        throw new Error('aborted');
      }
      return {
        stream: partsStream([]),
        rawCall: { rawPrompt: '', rawSettings: {} },
      } as any;
    });
    const model = new AiSdkModel(
      createMockModel({
        doStream,
      }),
    );

    await expect(async () => {
      const iter = model.getStreamedResponse({
        input: 'hi',
        tools: [],
        handoffs: [],
        modelSettings: {},
        outputType: 'text',
        tracing: false,
        signal: abort.signal,
      } as any);
      for await (const _ of iter) {
        /* nothing */
      }
    }).rejects.toThrow('aborted');
    expect(doStream).toHaveBeenCalled();
  });

  test('prepends system instructions to prompt for doStream', async () => {
    let received: any;
    const model = new AiSdkModel(
      createMockModel({
        doStream: async (options) => {
          received = options.prompt;
          return {
            stream: simulateReadableStream({
              chunks: [],
              chunkDelayInMs: 0,
            }),
            rawCall: { rawPrompt: '', rawSettings: {} },
          };
        },
      }),
    );

    const iter = model.getStreamedResponse({
      systemInstructions: 'inst',
      input: 'hi',
      tools: [],
      handoffs: [],
      modelSettings: {},
      outputType: 'text',
      tracing: false,
    } as any);

    for await (const _ of iter) {
      // exhaust iterator
    }

    expect(received[0]).toEqual({
      role: 'system',
      content: 'inst',
    });
  });

  test('handles NaN usage in stream finish event', async () => {
    const parts = [
      { type: 'text-delta', delta: 'a' },
      {
        type: 'finish',
        finishReason: 'stop',
        usage: { inputTokens: Number.NaN, outputTokens: Number.NaN },
      },
    ];
    const model = new AiSdkModel(createStreamingMock(parts));

    let final: any;
    for await (const ev of model.getStreamedResponse({
      input: 'hi',
      tools: [],
      handoffs: [],
      modelSettings: {},
      outputType: 'text',
      tracing: false,
    } as any)) {
      if (ev.type === 'response_done') {
        final = ev.response.usage;
      }
    }

    expect(final).toEqual({ inputTokens: 0, outputTokens: 0, totalTokens: 0 });
  });
});

describe('AiSdkModel', () => {
  test('should be available', () => {
    const model = new AiSdkModel({} as any);
    expect(model).toBeDefined();
  });

  test('converts trailing function_call items to messages', async () => {
    let received: any;
    const fakeModel = {
      specificationVersion: 'v1',
      provider: 'fake',
      modelId: 'm',
      defaultObjectGenerationMode: undefined,
      doGenerate: vi.fn(async (opts: any) => {
        received = opts.prompt;
        return {
          content: [{ type: 'text', text: 'ok' }],
          finishReason: 'stop',
          usage: { inputTokens: 0, outputTokens: 0 },
          providerOptions: {},
        };
      }),
    };

    const model = new AiSdkModel(fakeModel as any);
    await withTrace('t', () =>
      model.getResponse({
        input: [
          {
            type: 'function_call',
            id: '1',
            callId: 'call1',
            name: 'do',
            arguments: '{}',
            status: 'completed',
            providerData: { meta: 1 },
          } as protocol.FunctionCallItem,
        ],
        tools: [],
        handoffs: [],
        modelSettings: {},
        outputType: 'text',
        tracing: false,
      } as any),
    );

    expect(received).toEqual([
      {
        role: 'assistant',
        content: [
          {
            type: 'tool-call',
            toolCallId: 'call1',
            toolName: 'do',
            input: {},

            providerOptions: { meta: 1 },
          },
        ],
        providerOptions: { meta: 1 },
      },
    ]);
  });

  describe('parseArguments', () => {
    test('should parse valid JSON', () => {
      expect(parseArguments(undefined)).toEqual({});
      expect(parseArguments(null)).toEqual({});
      expect(parseArguments('')).toEqual({});
      expect(parseArguments(' ')).toEqual({});
      expect(parseArguments('{ ')).toEqual({});
      expect(parseArguments('foo')).toEqual({});
      expect(parseArguments('{}')).toEqual({});
      expect(parseArguments('{ }')).toEqual({});

      expect(parseArguments('"foo"')).toEqual('foo');
      expect(parseArguments('[]')).toEqual([]);
      expect(parseArguments('[1,2,3]')).toEqual([1, 2, 3]);
      expect(parseArguments('{"a":1}')).toEqual({ a: 1 });
      expect(parseArguments('{"a":1,"b":"c"}')).toEqual({ a: 1, b: 'c' });
    });
  });

  describe('toolChoice support', () => {
    test('passes toolChoice to AI SDK request', async () => {
      let receivedRequest: any;
      const model = new AiSdkModel(
        createMockModel({
          doGenerate: async (request) => {
            receivedRequest = request;
            return {
              content: [{ type: 'text', text: 'ok' }],
              finishReason: 'stop',
              usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
              providerMetadata: {},
              rawCall: { rawPrompt: '', rawSettings: {} },
              warnings: [],
            };
          },
        }),
      );

      await withTrace('t', () =>
        model.getResponse({
          input: 'hi',
          tools: [],
          handoffs: [],
          modelSettings: { toolChoice: 'required' },
          outputType: 'text',
          tracing: false,
        } as any),
      );

      expect(receivedRequest.toolChoice).toBe('required');
    });

    test('converts specific tool name toolChoice', async () => {
      let receivedRequest: any;
      const model = new AiSdkModel(
        createMockModel({
          doGenerate: async (request) => {
            receivedRequest = request;
            return {
              content: [{ type: 'text', text: 'ok' }],
              finishReason: 'stop',
              usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
              providerMetadata: {},
              rawCall: { rawPrompt: '', rawSettings: {} },
              warnings: [],
            };
          },
        }),
      );

      await withTrace('t', () =>
        model.getResponse({
          input: 'hi',
          tools: [],
          handoffs: [],
          modelSettings: { toolChoice: 'weather' },
          outputType: 'text',
          tracing: false,
        } as any),
      );

      expect(receivedRequest.toolChoice).toEqual({
        type: 'tool',
        toolName: 'weather',
      });
    });
  });
});
