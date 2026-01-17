<script lang="ts">
	interface Props {
		value: number; // Current value
		max?: number; // Maximum value (default 100 for percentage)
		label: string; // Label below value (e.g., "Memory", "CPU")
		unit?: string; // Unit suffix (e.g., "%", "MB")
		size?: 'sm' | 'md'; // Size variant
		thresholds?: { warning: number; danger: number };
	}

	let {
		value,
		max = 100,
		label,
		unit = '%',
		size = 'md',
		thresholds = { warning: 75, danger: 90 }
	}: Props = $props();

	const percentage = $derived(Math.min(100, Math.max(0, (value / max) * 100)));
	const circumference = 2 * Math.PI * 40; // r=40
	const offset = $derived(circumference - (percentage / 100) * circumference);

	const sizes = { sm: 56, md: 72 };
	const strokeWidths = { sm: 5, md: 6 };

	const colorClass = $derived(
		percentage >= thresholds.danger
			? 'text-red-500'
			: percentage >= thresholds.warning
				? 'text-yellow-500'
				: 'text-green-500'
	);
</script>

<div
	class="relative flex items-center justify-center"
	style="width: {sizes[size]}px; height: {sizes[size]}px;"
>
	<svg width={sizes[size]} height={sizes[size]} viewBox="0 0 100 100" class="transform -rotate-90">
		<!-- Background circle -->
		<circle
			cx="50"
			cy="50"
			r="40"
			stroke="currentColor"
			stroke-width={strokeWidths[size]}
			fill="none"
			class="text-gray-200 dark:text-gray-700"
		/>
		<!-- Progress circle -->
		<circle
			cx="50"
			cy="50"
			r="40"
			stroke="currentColor"
			stroke-width={strokeWidths[size]}
			fill="none"
			stroke-dasharray={circumference}
			stroke-dashoffset={offset}
			stroke-linecap="round"
			class="{colorClass} transition-all duration-300"
		/>
	</svg>
	<div class="absolute text-center">
		<span class="text-sm font-semibold">{value.toFixed(0)}{unit}</span>
		<span class="block text-[10px] text-muted-foreground">{label}</span>
	</div>
</div>
