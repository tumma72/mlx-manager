<script lang="ts">
	import { Combobox } from 'bits-ui';
	import type { ServerProfile } from '$api';
	import { Button } from '$components/ui';
	import { Play, ChevronDown, Loader2 } from 'lucide-svelte';

	interface Props {
		profiles: ServerProfile[];
		onStart: (profile: ServerProfile) => Promise<void>;
		disabled?: boolean;
	}

	let { profiles, onStart, disabled = false }: Props = $props();

	let selectedValue = $state<string>('');
	let searchValue = $state('');
	let open = $state(false);
	let starting = $state(false);

	// Stabilize profiles array: only update reference when IDs actually change.
	// This prevents the Combobox from resetting its internal state on every poll.
	let stableProfiles = $state<ServerProfile[]>([]);
	let lastProfileIds = $state<string>('');

	$effect(() => {
		const currentIds = profiles.map((p) => p.id).join(',');
		if (currentIds !== lastProfileIds) {
			// Profile list actually changed - update our stable copy
			stableProfiles = [...profiles];
			lastProfileIds = currentIds;

			// Clear selection if the selected profile is no longer available
			if (selectedValue && !profiles.find((p) => p.id?.toString() === selectedValue)) {
				selectedValue = '';
			}
		}
	});

	const filteredProfiles = $derived(
		searchValue === ''
			? stableProfiles
			: stableProfiles.filter(
					(p) =>
						p.name.toLowerCase().includes(searchValue.toLowerCase()) ||
						p.model_path.toLowerCase().includes(searchValue.toLowerCase())
				)
	);

	const selectedProfile = $derived(stableProfiles.find((p) => p.id?.toString() === selectedValue));

	async function handleStart() {
		if (selectedProfile && !starting) {
			starting = true;
			try {
				await onStart(selectedProfile);
				// Clear selection after successful start
				selectedValue = '';
				searchValue = '';
			} finally {
				starting = false;
			}
		}
	}

	function handleOpenChange(isOpen: boolean) {
		open = isOpen;
		if (!isOpen) {
			// Clear search when closing without selection
			searchValue = '';
		}
	}
</script>

<div class="flex gap-2">
	<Combobox.Root
		type="single"
		bind:value={selectedValue}
		bind:open
		onOpenChange={handleOpenChange}
		{disabled}
	>
		<div class="relative flex-1">
			<Combobox.Input
				class="h-10 w-full rounded-md border border-input bg-background px-3 py-2 pr-8 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
				placeholder="Select profile to start..."
				oninput={(e) => (searchValue = e.currentTarget.value)}
			/>
			<Combobox.Trigger
				class="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
			>
				<ChevronDown class="h-4 w-4" />
			</Combobox.Trigger>
		</div>

		<Combobox.Portal>
			<Combobox.Content
				class="z-50 max-h-60 overflow-auto rounded-md border bg-popover p-1 shadow-md"
				sideOffset={4}
			>
				{#each filteredProfiles as profile (profile.id)}
					<Combobox.Item
						value={profile.id?.toString() ?? ''}
						label={profile.name}
						class="flex cursor-pointer flex-col gap-0.5 rounded px-2 py-1.5 text-sm data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground"
					>
						<span class="font-medium">{profile.name}</span>
						<span class="truncate text-xs text-muted-foreground">{profile.model_path}</span>
					</Combobox.Item>
				{/each}
				{#if filteredProfiles.length === 0}
					<div class="px-2 py-4 text-center text-sm text-muted-foreground">No profiles found</div>
				{/if}
			</Combobox.Content>
		</Combobox.Portal>
	</Combobox.Root>

	<Button onclick={handleStart} disabled={!selectedProfile || starting || disabled}>
		{#if starting}
			<Loader2 class="mr-1 h-4 w-4 animate-spin" />
			Starting...
		{:else}
			<Play class="mr-1 h-4 w-4" />
			Start
		{/if}
	</Button>
</div>
