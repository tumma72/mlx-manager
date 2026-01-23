<script lang="ts">
  import { Collapsible } from 'bits-ui';
  import { ChevronDown, ChevronRight, AlertCircle, Copy, Check } from 'lucide-svelte';

  interface Props {
    summary: string;
    details?: string;
    defaultExpanded?: boolean;
  }

  let { summary, details, defaultExpanded = true }: Props = $props();

  let expanded = $state(defaultExpanded);
  let copied = $state(false);

  async function copyToClipboard() {
    const text = details ? `${summary}\n\n${details}` : summary;
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
      setTimeout(() => { copied = false; }, 2000);
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      copied = true;
      setTimeout(() => { copied = false; }, 2000);
    }
  }

  // Collapse this error (called from parent when new message sent)
  export function collapse() {
    expanded = false;
  }
</script>

<div class="rounded-lg border border-destructive/50 bg-destructive/10 p-3">
  <div class="flex items-start gap-2">
    <AlertCircle class="w-4 h-4 text-destructive mt-0.5 flex-shrink-0" />

    <div class="flex-1 min-w-0">
      {#if details}
        <Collapsible.Root bind:open={expanded}>
          <div class="flex items-center justify-between gap-2">
            <Collapsible.Trigger class="flex items-center gap-1 text-sm font-medium text-destructive hover:underline">
              {#if expanded}
                <ChevronDown class="w-4 h-4" />
              {:else}
                <ChevronRight class="w-4 h-4" />
              {/if}
              {summary}
            </Collapsible.Trigger>

            <button
              type="button"
              onclick={copyToClipboard}
              class="p-1 rounded hover:bg-destructive/20 transition-colors"
              title="Copy error"
            >
              {#if copied}
                <Check class="w-4 h-4 text-green-600" />
              {:else}
                <Copy class="w-4 h-4 text-destructive" />
              {/if}
            </button>
          </div>

          <Collapsible.Content>
            <pre class="mt-2 p-2 text-xs text-destructive bg-destructive/5 rounded overflow-x-auto whitespace-pre-wrap break-words">{details}</pre>
          </Collapsible.Content>
        </Collapsible.Root>
      {:else}
        <div class="flex items-center justify-between gap-2">
          <span class="text-sm font-medium text-destructive">{summary}</span>

          <button
            type="button"
            onclick={copyToClipboard}
            class="p-1 rounded hover:bg-destructive/20 transition-colors"
            title="Copy error"
          >
            {#if copied}
              <Check class="w-4 h-4 text-green-600" />
            {:else}
              <Copy class="w-4 h-4 text-destructive" />
            {/if}
          </button>
        </div>
      {/if}
    </div>
  </div>
</div>
