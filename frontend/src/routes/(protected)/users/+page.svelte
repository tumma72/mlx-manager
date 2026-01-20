<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { authStore } from '$stores';
	import { auth } from '$lib/api/client';
	import type { User, UserStatus } from '$lib/api/types';
	import { Card, Button, Badge, Input, ConfirmDialog } from '$components/ui';
	import { Loader2, Shield, ShieldOff, Trash2, KeyRound, UserCheck, UserX } from 'lucide-svelte';

	// State
	let users = $state<User[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let actionInProgress = $state<number | null>(null);

	// Reset password modal state
	let resetPasswordModal = $state<{ userId: number; email: string } | null>(null);
	let newPassword = $state('');
	let resetPasswordError = $state<string | null>(null);
	let resetPasswordLoading = $state(false);

	// Delete confirmation dialog
	let deleteConfirmOpen = $state(false);
	let userToDelete = $state<User | null>(null);

	// Admin-only guard
	onMount(() => {
		if (!authStore.isAdmin) {
			goto(resolve('/'));
			return;
		}
		loadUsers();
	});

	async function loadUsers() {
		loading = true;
		error = null;
		try {
			const result = await auth.listUsers();
			// Sort: pending first, then by created_at desc
			users = result.sort((a, b) => {
				if (a.status === 'pending' && b.status !== 'pending') return -1;
				if (b.status === 'pending' && a.status !== 'pending') return 1;
				return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
			});
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load users';
		} finally {
			loading = false;
		}
	}

	async function updateUserStatus(userId: number, status: UserStatus) {
		actionInProgress = userId;
		try {
			await auth.updateUser(userId, { status });
			await loadUsers();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to update user';
		} finally {
			actionInProgress = null;
		}
	}

	async function toggleAdmin(userId: number, makeAdmin: boolean) {
		actionInProgress = userId;
		try {
			await auth.updateUser(userId, { is_admin: makeAdmin });
			await loadUsers();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to update user';
		} finally {
			actionInProgress = null;
		}
	}

	function openDeleteConfirm(user: User) {
		userToDelete = user;
		deleteConfirmOpen = true;
	}

	async function confirmDelete() {
		if (!userToDelete) return;
		actionInProgress = userToDelete.id;
		try {
			await auth.deleteUser(userToDelete.id);
			await loadUsers();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete user';
		} finally {
			actionInProgress = null;
			userToDelete = null;
		}
	}

	function openResetPasswordModal(user: User) {
		resetPasswordModal = { userId: user.id, email: user.email };
		newPassword = '';
		resetPasswordError = null;
	}

	async function submitResetPassword() {
		if (!resetPasswordModal) return;
		if (newPassword.length < 8) {
			resetPasswordError = 'Password must be at least 8 characters';
			return;
		}
		resetPasswordLoading = true;
		resetPasswordError = null;
		try {
			await auth.resetPassword(resetPasswordModal.userId, newPassword);
			resetPasswordModal = null;
			newPassword = '';
		} catch (e) {
			resetPasswordError = e instanceof Error ? e.message : 'Failed to reset password';
		} finally {
			resetPasswordLoading = false;
		}
	}

	function getStatusBadgeVariant(status: UserStatus): 'warning' | 'success' | 'destructive' {
		switch (status) {
			case 'pending':
				return 'warning';
			case 'approved':
				return 'success';
			case 'disabled':
				return 'destructive';
		}
	}

	// Count admins (for self-protection)
	const adminCount = $derived(users.filter((u) => u.is_admin).length);

	// Helper to check if action should be disabled (self-protection)
	function isOnlyAdmin(user: User): boolean {
		return user.is_admin && adminCount === 1;
	}

	function isSelf(user: User): boolean {
		return authStore.user?.id === user.id;
	}

	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleDateString(undefined, {
			year: 'numeric',
			month: 'short',
			day: 'numeric'
		});
	}
</script>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<h1 class="text-2xl font-bold">User Management</h1>
	</div>

	{#if loading}
		<div class="text-center py-12 text-muted-foreground">
			<Loader2 class="w-6 h-6 animate-spin mx-auto mb-2" />
			Loading users...
		</div>
	{:else if error}
		<div class="text-center py-12">
			<p class="text-red-500 dark:text-red-400 mb-4">{error}</p>
			<Button onclick={() => loadUsers()}>Retry</Button>
		</div>
	{:else if users.length === 0}
		<Card class="p-12 text-center">
			<p class="text-muted-foreground">No users found.</p>
		</Card>
	{:else}
		<Card class="overflow-hidden">
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead class="bg-muted/50 border-b">
						<tr>
							<th class="text-left py-3 px-4 font-medium">Email</th>
							<th class="text-left py-3 px-4 font-medium">Status</th>
							<th class="text-left py-3 px-4 font-medium">Role</th>
							<th class="text-left py-3 px-4 font-medium">Created</th>
							<th class="text-right py-3 px-4 font-medium">Actions</th>
						</tr>
					</thead>
					<tbody class="divide-y">
						{#each users as user (user.id)}
							{@const isProcessing = actionInProgress === user.id}
							<tr class="hover:bg-muted/30 transition-colors">
								<td class="py-3 px-4">
									<div class="flex items-center gap-2">
										{user.email}
										{#if isSelf(user)}
											<span class="text-xs text-muted-foreground">(you)</span>
										{/if}
									</div>
								</td>
								<td class="py-3 px-4">
									<Badge variant={getStatusBadgeVariant(user.status)}>
										{user.status}
									</Badge>
								</td>
								<td class="py-3 px-4">
									{#if user.is_admin}
										<Badge variant="default">Admin</Badge>
									{:else}
										<span class="text-muted-foreground">User</span>
									{/if}
								</td>
								<td class="py-3 px-4 text-muted-foreground">
									{formatDate(user.created_at)}
								</td>
								<td class="py-3 px-4">
									<div class="flex items-center justify-end gap-2">
										{#if isProcessing}
											<Loader2 class="w-4 h-4 animate-spin" />
										{:else}
											<!-- Status actions -->
											{#if user.status === 'pending'}
												<Button
													size="sm"
													variant="outline"
													onclick={() => updateUserStatus(user.id, 'approved')}
													title="Approve user"
												>
													<UserCheck class="w-4 h-4 mr-1" />
													Approve
												</Button>
											{:else if user.status === 'approved'}
												{#if !isSelf(user) || !isOnlyAdmin(user)}
													<Button
														size="sm"
														variant="ghost"
														onclick={() => updateUserStatus(user.id, 'disabled')}
														disabled={isSelf(user) && isOnlyAdmin(user)}
														title={isSelf(user) && isOnlyAdmin(user)
															? 'Cannot disable last admin'
															: 'Disable user'}
													>
														<UserX class="w-4 h-4" />
													</Button>
												{:else}
													<Button
														size="sm"
														variant="ghost"
														disabled
														title="Cannot disable last admin"
													>
														<UserX class="w-4 h-4 opacity-50" />
													</Button>
												{/if}
											{:else if user.status === 'disabled'}
												<Button
													size="sm"
													variant="ghost"
													onclick={() => updateUserStatus(user.id, 'approved')}
													title="Enable user"
												>
													<UserCheck class="w-4 h-4" />
												</Button>
											{/if}

											<!-- Admin toggle -->
											{#if !user.is_admin}
												<Button
													size="sm"
													variant="ghost"
													onclick={() => toggleAdmin(user.id, true)}
													title="Make admin"
												>
													<Shield class="w-4 h-4" />
												</Button>
											{:else if !isSelf(user) || !isOnlyAdmin(user)}
												<Button
													size="sm"
													variant="ghost"
													onclick={() => toggleAdmin(user.id, false)}
													disabled={isOnlyAdmin(user)}
													title={isOnlyAdmin(user)
														? 'Cannot remove last admin'
														: 'Remove admin'}
												>
													<ShieldOff class="w-4 h-4" />
												</Button>
											{:else}
												<Button
													size="sm"
													variant="ghost"
													disabled
													title="Cannot remove last admin"
												>
													<ShieldOff class="w-4 h-4 opacity-50" />
												</Button>
											{/if}

											<!-- Reset password -->
											<Button
												size="sm"
												variant="ghost"
												onclick={() => openResetPasswordModal(user)}
												title="Reset password"
											>
												<KeyRound class="w-4 h-4" />
											</Button>

											<!-- Delete -->
											{#if !isSelf(user) || !isOnlyAdmin(user)}
												<Button
													size="sm"
													variant="ghost"
													onclick={() => openDeleteConfirm(user)}
													disabled={isSelf(user) && isOnlyAdmin(user)}
													title={isSelf(user) && isOnlyAdmin(user)
														? 'Cannot delete last admin'
														: 'Delete user'}
												>
													<Trash2 class="w-4 h-4 text-destructive" />
												</Button>
											{:else}
												<Button
													size="sm"
													variant="ghost"
													disabled
													title="Cannot delete last admin"
												>
													<Trash2 class="w-4 h-4 opacity-50" />
												</Button>
											{/if}
										{/if}
									</div>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</Card>
	{/if}
</div>

<!-- Reset Password Modal -->
{#if resetPasswordModal}
	<div class="fixed inset-0 z-50 bg-black/50 flex items-center justify-center">
		<div class="bg-background border rounded-lg shadow-lg p-6 w-full max-w-md">
			<h2 class="text-lg font-semibold mb-4">Reset Password</h2>
			<p class="text-sm text-muted-foreground mb-4">
				Set a new password for <strong>{resetPasswordModal.email}</strong>
			</p>

			<form
				onsubmit={(e) => {
					e.preventDefault();
					submitResetPassword();
				}}
			>
				<div class="space-y-4">
					<div>
						<label for="new-password" class="block text-sm font-medium mb-1">New Password</label>
						<Input
							id="new-password"
							type="password"
							bind:value={newPassword}
							placeholder="Enter new password (min 8 characters)"
							required
						/>
					</div>

					{#if resetPasswordError}
						<p class="text-sm text-red-500">{resetPasswordError}</p>
					{/if}

					<div class="flex justify-end gap-2">
						<Button
							variant="outline"
							onclick={() => {
								resetPasswordModal = null;
								newPassword = '';
							}}
							disabled={resetPasswordLoading}
						>
							Cancel
						</Button>
						<Button type="submit" disabled={resetPasswordLoading || newPassword.length < 8}>
							{#if resetPasswordLoading}
								<Loader2 class="w-4 h-4 animate-spin mr-2" />
							{/if}
							Reset Password
						</Button>
					</div>
				</div>
			</form>
		</div>
	</div>
{/if}

<!-- Delete Confirmation Dialog -->
<ConfirmDialog
	bind:open={deleteConfirmOpen}
	title="Delete User"
	description={`Are you sure you want to delete ${userToDelete?.email}? This action cannot be undone.`}
	confirmLabel="Delete"
	variant="destructive"
	onConfirm={confirmDelete}
	onCancel={() => (userToDelete = null)}
/>
