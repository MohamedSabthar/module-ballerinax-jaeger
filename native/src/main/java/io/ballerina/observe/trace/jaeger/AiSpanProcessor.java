package io.ballerina.observe.trace.jaeger;

import com.google.gson.*;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.context.Context;
import io.opentelemetry.sdk.common.InstrumentationLibraryInfo;
import io.opentelemetry.sdk.trace.ReadWriteSpan;
import io.opentelemetry.sdk.trace.ReadableSpan;
import io.opentelemetry.sdk.trace.SpanProcessor;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.data.StatusData;
import io.opentelemetry.sdk.trace.export.SpanExporter;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * SpanProcessor that transforms GenAI spans to strict OpenInference convention.
 * Only exports spans with OpenInference-compliant attributes and naming.
 */
public class AiSpanProcessor implements SpanProcessor {

    private final SpanExporter delegate;
    private final Map<String, ReadableSpan> spanCache = new ConcurrentHashMap<>();
    private static final Gson gson = new Gson();

    // Original attribute
    private static final AttributeKey<String> SPAN_TYPE = AttributeKey.stringKey("span.type");

    // GenAI semantic convention attributes (source)
    private static final AttributeKey<String> GEN_AI_OPERATION_NAME =
            AttributeKey.stringKey("gen_ai.operation.name");
    private static final AttributeKey<String> GEN_AI_PROVIDER_NAME =
            AttributeKey.stringKey("gen_ai.provider.name");
    private static final AttributeKey<String> GEN_AI_CONVERSATION_ID =
            AttributeKey.stringKey("gen_ai.conversation.id");
    private static final AttributeKey<String> GEN_AI_REQUEST_MODEL =
            AttributeKey.stringKey("gen_ai.request.model");
    private static final AttributeKey<String> GEN_AI_RESPONSE_MODEL =
            AttributeKey.stringKey("gen_ai.response.model");
    private static final AttributeKey<String> GEN_AI_TEMPERATURE =
            AttributeKey.stringKey("gen_ai.request.temperature");
    private static final AttributeKey<String> GEN_AI_STOP_SEQUENCE =
            AttributeKey.stringKey("gen_ai.request.stop_sequences");
    private static final AttributeKey<String> GEN_AI_FINISH_REASON =
            AttributeKey.stringKey("gen_ai.response.finish_reasons");
    private static final AttributeKey<String> GEN_AI_INPUT_TOKENS =
            AttributeKey.stringKey("gen_ai.usage.input_tokens");
    private static final AttributeKey<String> GEN_AI_OUTPUT_TOKENS =
            AttributeKey.stringKey("gen_ai.usage.output_tokens");
    private static final AttributeKey<String> GEN_AI_INPUT_MESSAGES =
            AttributeKey.stringKey("gen_ai.input.messages");
    private static final AttributeKey<String> GEN_AI_OUTPUT_MESSAGES =
            AttributeKey.stringKey("gen_ai.output.messages");
    private static final AttributeKey<String> GEN_AI_SYSTEM_INSTRUCTIONS =
            AttributeKey.stringKey("gen_ai.system_instructions");
    private static final AttributeKey<String> GEN_AI_INPUT_CONTENT =
            AttributeKey.stringKey("gen_ai.input.content");
    private static final AttributeKey<String> GEN_AI_INPUT_TOOLS =
            AttributeKey.stringKey("gen_ai.input.tools");
    private static final AttributeKey<String> GEN_AI_AGENT_NAME =
            AttributeKey.stringKey("gen_ai.agent.name");
    private static final AttributeKey<String> GEN_AI_TOOL_NAME =
            AttributeKey.stringKey("gen_ai.tool.name");
    private static final AttributeKey<String> GEN_AI_TOOL_DESCRIPTION =
            AttributeKey.stringKey("gen_ai.tool.description");
    private static final AttributeKey<String> GEN_AI_TOOL_ARGUMENTS =
            AttributeKey.stringKey("gen_ai.tool.arguments");
    private static final AttributeKey<String> GEN_AI_KB_RETRIEVE_INPUT_QUERY =
            AttributeKey.stringKey("gen_ai.knowledge_base.retrieve.input.query");
    private static final AttributeKey<String> GEN_AI_KB_NAME =
            AttributeKey.stringKey("gen_ai.knowledge_base.name");

    public AiSpanProcessor(SpanExporter delegate) {
        this.delegate = delegate;
    }

    @Override
    public void onStart(Context parentContext, ReadWriteSpan span) {
        String spanId = span.getSpanContext().getSpanId();
        spanCache.put(spanId, span);
    }

    @Override
    public void onEnd(ReadableSpan span) {
        SpanData spanData = span.toSpanData();

        // Only process AI spans
        if (!"ai".equals(spanData.getAttributes().get(SPAN_TYPE))) {
            spanCache.remove(span.getSpanContext().getSpanId());
            return;
        }

        // Transform span to OpenInference convention
        SpanData transformedSpan = transformToOpenInference(spanData);

        if (transformedSpan == null) {
            // Skip spans that can't be transformed
            spanCache.remove(span.getSpanContext().getSpanId());
            return;
        }

        // Reparent if needed
        String parentSpanId = transformedSpan.getParentSpanContext().getSpanId();

        if (parentSpanId == null || parentSpanId.isEmpty() ||
                !transformedSpan.getParentSpanContext().isValid()) {
            delegate.export(List.of(transformedSpan));
        } else {
            ReadableSpan parentSpan = spanCache.get(parentSpanId);

            if (parentSpan != null && isAiSpan(parentSpan)) {
                delegate.export(List.of(transformedSpan));
            } else {
                String aiAncestorId = findNearestAiAncestor(parentSpanId);

                if (aiAncestorId != null) {
                    SpanData reparentedSpan = reparentSpan(transformedSpan, aiAncestorId);
                    delegate.export(List.of(reparentedSpan));
                } else {
                    SpanData rootSpan = makeRootSpan(transformedSpan);
                    delegate.export(List.of(rootSpan));
                }
            }
        }

        spanCache.remove(span.getSpanContext().getSpanId());
    }

    /**
     * Transform GenAI span to OpenInference convention with proper naming and attributes.
     */
    private SpanData transformToOpenInference(SpanData original) {
        Attributes originalAttrs = original.getAttributes();
        String operationName = originalAttrs.get(GEN_AI_OPERATION_NAME);

        if (operationName == null) {
            return null; // Can't transform without operation
        }

        String newSpanName;
        Attributes newAttributes;

        switch (operationName) {
            case "chat":
            case "generate_content":
                newSpanName = buildLLMSpanName(originalAttrs);
                newAttributes = buildLLMAttributes(originalAttrs);
                break;
            case "embeddings":
                newSpanName = buildEmbeddingSpanName(originalAttrs);
                newAttributes = buildEmbeddingAttributes(originalAttrs);
                break;
            case "knowledge_base_retrieve":
                newSpanName = "Retriever";
                newAttributes = buildRetrieverAttributes(originalAttrs);
                break;
            case "execute_tool":
                newSpanName = buildToolSpanName(originalAttrs);
                newAttributes = buildToolAttributes(originalAttrs);
                break;
            case "invoke_agent":
            case "create_agent":
                newSpanName = buildAgentSpanName(originalAttrs);
                newAttributes = buildAgentAttributes(originalAttrs);
                break;
            case "create_knowledge_base":
            case "knowledge_base_ingest":
                newSpanName = buildChainSpanName(originalAttrs, operationName);
                newAttributes = buildChainAttributes(originalAttrs);
                break;
            default:
                newSpanName = operationName;
                newAttributes = buildChainAttributes(originalAttrs);
                break;
        }

        return new TransformedSpanData(original, newSpanName, newAttributes);
    }

    // ============ Span Name Builders ============

    private String buildLLMSpanName(Attributes attrs) {
        String model = getFirstNonNull(
                attrs.get(GEN_AI_REQUEST_MODEL),
                attrs.get(GEN_AI_RESPONSE_MODEL)
        );
        return model != null ? "llm " + model : "llm";
    }

    private String buildEmbeddingSpanName(Attributes attrs) {
        String model = getFirstNonNull(
                attrs.get(GEN_AI_REQUEST_MODEL),
                attrs.get(GEN_AI_RESPONSE_MODEL)
        );
        return model != null ? "Embedding " + model : "Embedding";
    }

    private String buildToolSpanName(Attributes attrs) {
        String toolName = attrs.get(GEN_AI_TOOL_NAME);
        return toolName != null ? toolName : "tool";
    }

    private String buildAgentSpanName(Attributes attrs) {
        String agentName = attrs.get(GEN_AI_AGENT_NAME);
        return agentName != null ? agentName : "Agent";
    }

    private String buildChainSpanName(Attributes attrs, String operation) {
        String kbName = attrs.get(GEN_AI_KB_NAME);
        if (kbName != null) {
            return kbName;
        }
        // Convert operation name to readable format
        return operation.replace("_", " ");
    }

    // ============ Attribute Builders ============

    private Attributes buildLLMAttributes(Attributes original) {
        AttributesBuilder builder = Attributes.builder();

        // Required: span kind
        builder.put("openinference.span.kind", "LLM");

        // Model name (required for LLM)
        String model = getFirstNonNull(
                original.get(GEN_AI_REQUEST_MODEL),
                original.get(GEN_AI_RESPONSE_MODEL)
        );
        if (model != null) {
            builder.put("llm.model_name", model);
        }

        // Provider
        String provider = original.get(GEN_AI_PROVIDER_NAME);
        if (provider != null) {
            builder.put("llm.provider", provider);
        }

        // Input/Output messages - flatten if JSON
        String inputMessages = original.get(GEN_AI_INPUT_MESSAGES);
        if (inputMessages != null) {
            flattenMessages(builder, inputMessages, "llm.input_messages");
        }

        String outputMessages = original.get(GEN_AI_OUTPUT_MESSAGES);
        if (outputMessages != null) {
            flattenMessages(builder, outputMessages, "llm.output_messages");
        }

        // System prompt
        String systemInstructions = original.get(GEN_AI_SYSTEM_INSTRUCTIONS);
        if (systemInstructions != null) {
            builder.put("llm.system", systemInstructions);
        }

        // Invocation parameters
        String invocationParams = buildInvocationParameters(original);
        if (invocationParams != null) {
            builder.put("llm.invocation_parameters", invocationParams);
        }

        // Token counts
        Long inputTokens = original.get(GEN_AI_INPUT_TOKENS) == null ? null : Long.parseLong(Objects.requireNonNull(original.get(GEN_AI_INPUT_TOKENS)));
        Long outputTokens = original.get(GEN_AI_OUTPUT_TOKENS) == null ? null : Long.parseLong(Objects.requireNonNull(original.get(GEN_AI_OUTPUT_TOKENS)));

        if (inputTokens != null) {
            builder.put("llm.token_count.prompt", inputTokens);
        }
        if (outputTokens != null) {
            builder.put("llm.token_count.completion", outputTokens);
        }
        if (inputTokens != null && outputTokens != null) {
            builder.put("llm.token_count.total", inputTokens + outputTokens);
        }

        // Session ID from conversation ID
        String conversationId = original.get(GEN_AI_CONVERSATION_ID);
        if (conversationId != null) {
            builder.put("session.id", conversationId);
        }

        // Tools as JSON schema
        String tools = original.get(GEN_AI_INPUT_TOOLS);
        if (tools != null) {
            flattenTools(builder, tools);
        }

        return builder.build();
    }

    private Attributes buildEmbeddingAttributes(Attributes original) {
        AttributesBuilder builder = Attributes.builder();

        builder.put("openinference.span.kind", "EMBEDDING");

        // Model name
        String model = getFirstNonNull(
                original.get(GEN_AI_REQUEST_MODEL),
                original.get(GEN_AI_RESPONSE_MODEL)
        );
        if (model != null) {
            builder.put("embedding.model_name", model);
        }

        // Text to embed
        String text = getFirstNonNull(
                original.get(GEN_AI_INPUT_CONTENT),
                original.get(GEN_AI_INPUT_MESSAGES)
        );
        if (text != null) {
            builder.put("embedding.text", text);
        }

        return builder.build();
    }

    private Attributes buildRetrieverAttributes(Attributes original) {
        AttributesBuilder builder = Attributes.builder();

        builder.put("openinference.span.kind", "RETRIEVER");

        String query = original.get(GEN_AI_KB_RETRIEVE_INPUT_QUERY);
        if (query != null) {
            builder.put("retrieval.query", query);
        }

        return builder.build();
    }

    private Attributes buildToolAttributes(Attributes original) {
        AttributesBuilder builder = Attributes.builder();

        builder.put("openinference.span.kind", "TOOL");

        String toolName = original.get(GEN_AI_TOOL_NAME);
        if (toolName != null) {
            builder.put("tool.name", toolName);
        }

        String description = original.get(GEN_AI_TOOL_DESCRIPTION);
        if (description != null) {
            builder.put("tool.description", description);
        }

        String arguments = original.get(GEN_AI_TOOL_ARGUMENTS);
        if (arguments != null) {
            builder.put("tool.parameters", arguments);
        }

        return builder.build();
    }

    private Attributes buildAgentAttributes(Attributes original) {
        AttributesBuilder builder = Attributes.builder();

        builder.put("openinference.span.kind", "AGENT");

        String agentName = original.get(GEN_AI_AGENT_NAME);
        if (agentName != null) {
            builder.put("chain.name", agentName);
        }

        return builder.build();
    }

    private Attributes buildChainAttributes(Attributes original) {
        AttributesBuilder builder = Attributes.builder();

        builder.put("openinference.span.kind", "CHAIN");

        String kbName = original.get(GEN_AI_KB_NAME);
        if (kbName != null) {
            builder.put("chain.name", kbName);
        }

        return builder.build();
    }

    // ============ Helper Methods ============

    /**
     * Flatten JSON messages array to indexed attributes.
     * Example: llm.input_messages.0.message.role, llm.input_messages.0.message.content
     */
    private void flattenMessages(AttributesBuilder builder, String messagesJson, String prefix) {
        try {
            JsonElement element = JsonParser.parseString(messagesJson);
            if (element.isJsonArray()) {
                JsonArray array = element.getAsJsonArray();
                for (int i = 0; i < array.size(); i++) {
                    JsonObject msg = array.get(i).getAsJsonObject();

                    if (msg.has("role")) {
                        builder.put(prefix + "." + i + ".message.role",
                                msg.get("role").getAsString());
                    }
                    if (msg.has("content") && (msg.get("content") instanceof JsonPrimitive)) {
                        builder.put(prefix + "." + i + ".message.content",
                                msg.get("content").getAsString());
                    }
                    if (msg.has("content") && (msg.get("content") instanceof JsonArray)) {
                        builder.put(prefix + "." + i + ".message.content",
                                msg.get("content").getAsJsonArray().toString());
                    }
                    if (msg.has("content") && (msg.get("content") instanceof JsonObject)) {
                        builder.put(prefix + "." + i + ".message.content",
                                msg.get("content").getAsJsonObject().toString());
                    }
                    if (msg.has("toolCalls") && !(msg.get("toolCalls") instanceof JsonNull)) {
                        JsonArray toolCalls = msg.get("toolCalls").getAsJsonArray();
                        for (int j = 0; j < toolCalls.size(); j++) {
                            JsonObject toolCall = toolCalls.get(j).getAsJsonObject();
                            if (toolCall.has("id") && !(toolCall.get("id") instanceof JsonNull)) {
                                builder.put(prefix + "." + i + ".message.tool_calls." + j + ".tool_call.id", toolCall.get("id").getAsString());
                            }
                            if (toolCall.has("name") && !(toolCall.get("name") instanceof JsonNull)) {
                                builder.put(prefix + "." + i + ".message.tool_calls." + j + ".tool_call.function.name", toolCall.get("name").getAsString());
                            }
                            if (toolCall.has("arguments") && !(toolCall.get("arguments") instanceof JsonNull)) {
                                builder.put(prefix + "." + i + ".message.tool_calls." + j + ".tool_call.function.arguments", toolCall.get("arguments").getAsJsonObject().toString());
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            // If parsing fails, store as-is
            builder.put(prefix, messagesJson);
        }
    }

    /**
     * Flatten tools array to indexed JSON schema attributes.
     */
    private void flattenTools(AttributesBuilder builder, String toolsJson) {
        try {
            JsonElement element = JsonParser.parseString(toolsJson);
            if (element.isJsonArray()) {
                JsonArray array = element.getAsJsonArray();
                for (int i = 0; i < array.size(); i++) {
                    builder.put("llm.tools." + i + ".tool.json_schema",
                            array.get(i).toString());
                }
            }
        } catch (Exception e) {
            // Ignore if not valid JSON
        }
    }

    /**
     * Build invocation parameters JSON from temperature, stop sequences, etc.
     */
    private String buildInvocationParameters(Attributes attrs) {
        JsonObject params = new JsonObject();

        String temperature = attrs.get(GEN_AI_TEMPERATURE);
        if (temperature != null) {
            try {
                params.addProperty("temperature", Double.parseDouble(temperature));
            } catch (NumberFormatException e) {
                params.addProperty("temperature", temperature);
            }
        }

        String stopSeq = attrs.get(GEN_AI_STOP_SEQUENCE);
        if (stopSeq != null) {
            try {
                JsonElement stopElement = JsonParser.parseString(stopSeq);
                params.add("stop_sequences", stopElement);
            } catch (Exception e) {
                params.addProperty("stop_sequences", stopSeq);
            }
        }

        String finishReason = attrs.get(GEN_AI_FINISH_REASON);
        if (finishReason != null) {
            params.addProperty("finish_reason", finishReason);
        }

        return params.size() > 0 ? params.toString() : null;
    }

    private String getFirstNonNull(String... values) {
        for (String value : values) {
            if (value != null && !value.isEmpty()) {
                return value;
            }
        }
        return null;
    }

    private boolean isAiSpan(ReadableSpan span) {
        return "ai".equals(span.toSpanData().getAttributes().get(SPAN_TYPE));
    }

    private String findNearestAiAncestor(String currentSpanId) {
        if (currentSpanId == null || currentSpanId.isEmpty()) {
            return null;
        }

        ReadableSpan currentSpan = spanCache.get(currentSpanId);
        if (currentSpan == null) {
            return null;
        }

        if (isAiSpan(currentSpan)) {
            return currentSpanId;
        }

        String parentSpanId = currentSpan.toSpanData().getParentSpanContext().getSpanId();
        if (parentSpanId == null || parentSpanId.isEmpty() ||
                !currentSpan.toSpanData().getParentSpanContext().isValid()) {
            return null;
        }

        return findNearestAiAncestor(parentSpanId);
    }

    private SpanData reparentSpan(SpanData original, String newParentSpanId) {
        ReadableSpan newParentSpan = spanCache.get(newParentSpanId);

        if (newParentSpan == null) {
            return makeRootSpan(original);
        }

        SpanContext newParentContext = newParentSpan.getSpanContext();
        return new ReparentedSpanData(original, newParentContext);
    }

    private SpanData makeRootSpan(SpanData original) {
        SpanContext invalidParent = SpanContext.getInvalid();
        return new ReparentedSpanData(original, invalidParent);
    }

    @Override
    public boolean isStartRequired() {
        return true;
    }

    @Override
    public boolean isEndRequired() {
        return true;
    }

    /**
     * Wrapper that overrides span name and attributes with OpenInference conventions.
     */
    private static class TransformedSpanData implements SpanData {
        private final SpanData delegate;
        private final String newName;
        private final Attributes newAttributes;

        public TransformedSpanData(SpanData delegate, String newName, Attributes newAttributes) {
            this.delegate = delegate;
            this.newName = newName;
            this.newAttributes = newAttributes;
        }

        @Override
        public String getName() {
            return newName;
        }

        @Override
        public Attributes getAttributes() {
            return newAttributes;
        }

        @Override
        public io.opentelemetry.api.trace.SpanKind getKind() {
            return delegate.getKind();
        }

        @Override
        public SpanContext getSpanContext() {
            return delegate.getSpanContext();
        }

        @Override
        public SpanContext getParentSpanContext() {
            return delegate.getParentSpanContext();
        }

        @Override
        public StatusData getStatus() {
            return delegate.getStatus();
        }

        @Override
        public long getStartEpochNanos() {
            return delegate.getStartEpochNanos();
        }

        @Override
        public List<io.opentelemetry.sdk.trace.data.EventData> getEvents() {
            return delegate.getEvents();
        }

        @Override
        public List<io.opentelemetry.sdk.trace.data.LinkData> getLinks() {
            return delegate.getLinks();
        }

        @Override
        public long getEndEpochNanos() {
            return delegate.getEndEpochNanos();
        }

        @Override
        public boolean hasEnded() {
            return delegate.hasEnded();
        }

        @Override
        public int getTotalRecordedEvents() {
            return delegate.getTotalRecordedEvents();
        }

        @Override
        public int getTotalRecordedLinks() {
            return delegate.getTotalRecordedLinks();
        }

        @Override
        public int getTotalAttributeCount() {
            return newAttributes.size();
        }

        @Override
        public InstrumentationLibraryInfo getInstrumentationLibraryInfo() {
            return delegate.getInstrumentationLibraryInfo();
        }

        @Override
        public io.opentelemetry.sdk.resources.Resource getResource() {
            return delegate.getResource();
        }

        @Override
        public io.opentelemetry.sdk.common.InstrumentationScopeInfo getInstrumentationScopeInfo() {
            return delegate.getInstrumentationScopeInfo();
        }
    }

    /**
     * Wrapper that overrides parent context for reparenting.
     */
    private static class ReparentedSpanData implements SpanData {
        private final SpanData delegate;
        private final SpanContext newParentContext;

        public ReparentedSpanData(SpanData delegate, SpanContext newParentContext) {
            this.delegate = delegate;
            this.newParentContext = newParentContext;
        }

        @Override
        public SpanContext getParentSpanContext() {
            return newParentContext;
        }

        @Override
        public String getName() {
            return delegate.getName();
        }

        @Override
        public io.opentelemetry.api.trace.SpanKind getKind() {
            return delegate.getKind();
        }

        @Override
        public SpanContext getSpanContext() {
            return delegate.getSpanContext();
        }

        @Override
        public StatusData getStatus() {
            return delegate.getStatus();
        }

        @Override
        public long getStartEpochNanos() {
            return delegate.getStartEpochNanos();
        }

        @Override
        public Attributes getAttributes() {
            return delegate.getAttributes();
        }

        @Override
        public List<io.opentelemetry.sdk.trace.data.EventData> getEvents() {
            return delegate.getEvents();
        }

        @Override
        public List<io.opentelemetry.sdk.trace.data.LinkData> getLinks() {
            return delegate.getLinks();
        }

        @Override
        public long getEndEpochNanos() {
            return delegate.getEndEpochNanos();
        }

        @Override
        public boolean hasEnded() {
            return delegate.hasEnded();
        }

        @Override
        public int getTotalRecordedEvents() {
            return delegate.getTotalRecordedEvents();
        }

        @Override
        public int getTotalRecordedLinks() {
            return delegate.getTotalRecordedLinks();
        }

        @Override
        public int getTotalAttributeCount() {
            return delegate.getTotalAttributeCount();
        }

        @Override
        public InstrumentationLibraryInfo getInstrumentationLibraryInfo() {
            return delegate.getInstrumentationLibraryInfo();
        }

        @Override
        public io.opentelemetry.sdk.resources.Resource getResource() {
            return delegate.getResource();
        }

        @Override
        public io.opentelemetry.sdk.common.InstrumentationScopeInfo getInstrumentationScopeInfo() {
            return delegate.getInstrumentationScopeInfo();
        }
    }
}