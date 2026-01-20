<script lang="ts" module>
	export interface FilterState {
		architectures: string[];
		multimodal: boolean | null;
		quantization: number[];
	}

	export const ARCHITECTURE_OPTIONS = [
		'Llama',
		'Qwen',
		'Mistral',
		'Gemma',
		'Phi',
		'DeepSeek',
		'StarCoder',
		'GLM',
		'MiniMax'
	];

	export const QUANTIZATION_OPTIONS = [2, 3, 4, 8, 16];

	export function createEmptyFilters(): FilterState {
		return {
			architectures: [],
			multimodal: null,
			quantization: []
		};
	}
</script>

<script lang="ts">
	import { Dialog } from 'bits-ui';
	import { Button } from '$components/ui';
	import { X } from 'lucide-svelte';

	interface Props {
		open: boolean;
		filters: FilterState;
		onApply?: (filters: FilterState) => void;
	}

	let { open = $bindable(), filters = $bindable(), onApply }: Props = $props();

	// Local copy for editing until Apply
	let localFilters = $state<FilterState>(structuredClone(filters));

	// Reset local filters when modal opens
	$effect(() => {
		if (open) {
			localFilters = structuredClone(filters);
		}
	});

	function toggleArchitecture(arch: string) {
		const idx = localFilters.architectures.indexOf(arch);
		if (idx >= 0) {
			localFilters.architectures = localFilters.architectures.filter((a) => a !== arch);
		} else {
			localFilters.architectures = [...localFilters.architectures, arch];
		}
	}

	function toggleQuantization(bits: number) {
		const idx = localFilters.quantization.indexOf(bits);
		if (idx >= 0) {
			localFilters.quantization = localFilters.quantization.filter((q) => q !== bits);
		} else {
			localFilters.quantization = [...localFilters.quantization, bits];
		}
	}

	function setMultimodal(value: boolean | null) {
		localFilters.multimodal = value;
	}

	function clearAll() {
		localFilters = createEmptyFilters();
	}

	function applyFilters() {
		filters = structuredClone(localFilters);
		onApply?.(filters);
		open = false;
	}
</script>

<Dialog.Root bind:open>
	<Dialog.Portal>
		<Dialog.Overlay
			class="fixed inset-0 z-50 bg-black/50 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0"
		/>
		<Dialog.Content
			class="fixed left-[50%] top-[50%] z-50 w-full max-w-[400px] translate-x-[-50%] translate-y-[-50%] border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 sm:rounded-lg max-h-[80vh] overflow-y-auto"
		>
			<div class="flex items-center justify-between mb-4">
				<Dialog.Title class="text-lg font-semibold">Filter Models</Dialog.Title>
				<Dialog.Close
					class="rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
				>
					<X class="h-4 w-4" />
					<span class="sr-only">Close</span>
				</Dialog.Close>
			</div>

			<!-- Architecture section -->
			<section class="mb-6">
				<h4 class="text-sm font-medium mb-2">Architecture</h4>
				<div class="grid grid-cols-3 gap-2">
					{#each ARCHITECTURE_OPTIONS as arch (arch)}
						<label class="flex items-center gap-2 text-sm cursor-pointer">
							<input
								type="checkbox"
								checked={localFilters.architectures.includes(arch)}
								onchange={() => toggleArchitecture(arch)}
								class="rounded"
							/>
							{arch}
						</label>
					{/each}
				</div>
			</section>

			<!-- Capabilities section -->
			<section class="mb-6">
				<h4 class="text-sm font-medium mb-2">Capabilities</h4>
				<div class="space-y-2">
					<label class="flex items-center gap-2 text-sm cursor-pointer">
						<input
							type="radio"
							name="multimodal"
							checked={localFilters.multimodal === null}
							onchange={() => setMultimodal(null)}
						/>
						Any
					</label>
					<label class="flex items-center gap-2 text-sm cursor-pointer">
						<input
							type="radio"
							name="multimodal"
							checked={localFilters.multimodal === false}
							onchange={() => setMultimodal(false)}
						/>
						Text-only
					</label>
					<label class="flex items-center gap-2 text-sm cursor-pointer">
						<input
							type="radio"
							name="multimodal"
							checked={localFilters.multimodal === true}
							onchange={() => setMultimodal(true)}
						/>
						Multimodal (Vision)
					</label>
				</div>
			</section>

			<!-- Quantization section -->
			<section class="mb-6">
				<h4 class="text-sm font-medium mb-2">Quantization</h4>
				<div class="flex flex-wrap gap-3">
					{#each QUANTIZATION_OPTIONS as bits (bits)}
						<label class="flex items-center gap-2 text-sm cursor-pointer">
							<input
								type="checkbox"
								checked={localFilters.quantization.includes(bits)}
								onchange={() => toggleQuantization(bits)}
								class="rounded"
							/>
							{bits === 16 ? 'fp16' : `${bits}-bit`}
						</label>
					{/each}
				</div>
			</section>

			<div class="flex justify-between mt-6">
				<Button variant="ghost" onclick={clearAll}>Clear All</Button>
				<Button onclick={applyFilters}>Apply</Button>
			</div>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
