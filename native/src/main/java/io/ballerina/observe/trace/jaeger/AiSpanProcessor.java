package io.ballerina.observe.trace.jaeger;

import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
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
import java.util.concurrent.ConcurrentHashMap;

/**
 * SpanProcessor that filters and exports only AI spans, automatically
 * reparenting them to the nearest AI ancestor in the span hierarchy.
 */
public class AiSpanProcessor implements SpanProcessor {

    private final SpanExporter delegate;
    private final Map<String, ReadableSpan> spanCache = new ConcurrentHashMap<>();

    private static final AttributeKey<String> SPAN_TYPE = AttributeKey.stringKey("span.type");

    public AiSpanProcessor(SpanExporter delegate) {
        this.delegate = delegate;
    }

    @Override
    public void onStart(Context parentContext, ReadWriteSpan span) {
        // Cache all spans for parent lookup
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

        // Check if immediate parent is an AI span
        String parentSpanId = spanData.getParentSpanContext().getSpanId();

        if (parentSpanId == null || parentSpanId.isEmpty() || !spanData.getParentSpanContext().isValid()) {
            // Root span - export as is
            delegate.export(List.of(spanData));
        } else {
            ReadableSpan parentSpan = spanCache.get(parentSpanId);

            if (parentSpan != null && isAiSpan(parentSpan)) {
                // Immediate parent is AI - export as is
                delegate.export(List.of(spanData));
            } else {
                // Parent is not AI - find nearest AI ancestor recursively
                String aiAncestorId = findNearestAiAncestor(parentSpanId);

                if (aiAncestorId != null) {
                    // Reparent to AI ancestor
                    SpanData reparentedSpan = reparentSpan(spanData, aiAncestorId);
                    delegate.export(List.of(reparentedSpan));
                } else {
                    // No AI ancestor found - make it a root span
                    SpanData rootSpan = makeRootSpan(spanData);
                    delegate.export(List.of(rootSpan));
                }
            }
        }

        // Cleanup
        spanCache.remove(span.getSpanContext().getSpanId());
    }

    /**
     * Check if a span is an AI span.
     */
    private boolean isAiSpan(ReadableSpan span) {
        return "ai".equals(span.toSpanData().getAttributes().get(SPAN_TYPE));
    }

    /**
     * Recursively find the nearest AI ancestor span.
     * Returns the span ID of the nearest AI ancestor, or null if none found.
     */
    private String findNearestAiAncestor(String currentSpanId) {
        if (currentSpanId == null || currentSpanId.isEmpty()) {
            return null;
        }

        ReadableSpan currentSpan = spanCache.get(currentSpanId);
        if (currentSpan == null) {
            return null;
        }

        // Check if current span is an AI span
        if (isAiSpan(currentSpan)) {
            return currentSpanId;
        }

        // Recursively check parent
        String parentSpanId = currentSpan.toSpanData().getParentSpanContext().getSpanId();
        if (parentSpanId == null || parentSpanId.isEmpty() ||
                !currentSpan.toSpanData().getParentSpanContext().isValid()) {
            return null; // Reached root without finding AI span
        }

        return findNearestAiAncestor(parentSpanId);
    }

    /**
     * Create a new SpanData with a different parent.
     */
    private SpanData reparentSpan(SpanData original, String newParentSpanId) {
        ReadableSpan newParentSpan = spanCache.get(newParentSpanId);

        if (newParentSpan == null) {
            // Parent not in cache - make it a root span
            return makeRootSpan(original);
        }

        SpanContext newParentContext = newParentSpan.getSpanContext();
        return new ReparentedSpanData(original, newParentContext);
    }

    /**
     * Create a root span (no parent) from an existing span.
     */
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
     * Wrapper class that overrides the parent context while delegating
     * all other methods to the original SpanData.
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