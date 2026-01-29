# Phase 11: Configuration UI - Research

**Researched:** 2026-01-29
**Domain:** Configuration UI for model pool, cloud providers, and routing rules
**Confidence:** HIGH

## Summary

This phase implements a visual configuration interface for API keys (encrypted storage), model pool settings (memory, eviction), provider configuration (OpenAI/Anthropic), and model routing rules (pattern matching with drag-drop priority). The existing codebase provides a solid foundation: bits-ui components for UI primitives, SQLModel schemas for cloud credentials and backend mappings (added in Phase 10), and Svelte 5 runes-based stores.

Key research findings:
1. **bits-ui already provides needed components** - Accordion, Slider, Switch, Combobox are all available and already in use (v1.8.0 installed)
2. **API key encryption via Fernet** - Python's `cryptography` library provides Fernet symmetric encryption, ideal for storing API keys. Derive encryption key from existing `jwt_secret` using PBKDF2
3. **Drag-drop reordering** - `@rodrigodagostino/svelte-sortable-list` is Svelte 5 compatible (v2.x), provides keyboard accessibility and touch support
4. **Backend schemas exist** - `CloudCredential` and `BackendMapping` tables were added in Phase 10, ready for CRUD endpoints

**Primary recommendation:** Build a dedicated `/settings` route with three collapsible sections: Providers (accordion with API key inputs), Model Pool (slider + dropdown for memory/eviction), and Routing Rules (drag-drop cards with pattern inputs).

## Standard Stack

The established libraries/tools for this domain:

### Core (Already Installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| bits-ui | 1.8.0 | UI primitives (Accordion, Slider, Switch) | Already in project; headless, accessible |
| cryptography | 43+ | Fernet API key encryption | Industry standard; already used by pwdlib |
| lucide-svelte | 0.469+ | Icons (Key, Settings, Shield, etc.) | Already in project |

### New Dependencies
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|-------------|
| @rodrigodagostino/svelte-sortable-list | 2.x | Drag-drop rule reordering | Svelte 5 native, accessibility-first, zero deps |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| svelte-sortable-list | SortableJS | SortableJS needs wrapper; sortable-list is Svelte 5 native |
| Fernet | AWS KMS/HashiCorp Vault | Overkill for desktop app; Fernet is simpler |
| bits-ui Accordion | Custom collapsible | bits-ui provides accessibility free |

**Installation:**
```bash
# Frontend
npm install @rodrigodagostino/svelte-sortable-list

# Backend (cryptography already available via pwdlib[argon2])
# No new dependencies needed - cryptography is a transitive dependency
```

## Architecture Patterns

### Recommended Project Structure
```
frontend/src/
├── routes/(protected)/settings/
│   └── +page.svelte              # Main settings page
├── lib/components/settings/
│   ├── ProviderSection.svelte    # Accordion section for a provider
│   ├── ProviderForm.svelte       # API key input + connection test
│   ├── ModelPoolSettings.svelte  # Memory slider, eviction dropdown
│   ├── RoutingRulesSection.svelte# Drag-drop rule list
│   ├── RuleCard.svelte           # Individual rule card
│   ├── RuleTestInput.svelte      # Test which rule matches
│   └── index.ts
├── lib/stores/
│   └── settings.svelte.ts        # Configuration state management

backend/mlx_manager/
├── routers/
│   └── settings.py               # NEW: Settings CRUD endpoints
├── services/
│   └── encryption_service.py     # NEW: Fernet encryption for API keys
```

### Pattern 1: Accordion Provider Sections
**What:** Expandable sections for each cloud provider with status indicator
**When to use:** Provider configuration (OpenAI, Anthropic)
**Example:**
```svelte
<!-- Source: bits-ui Accordion docs -->
<script lang="ts">
  import { Accordion } from 'bits-ui';
  import { ChevronDown, CheckCircle, XCircle } from 'lucide-svelte';

  interface Provider {
    type: 'openai' | 'anthropic';
    configured: boolean;
    connectionStatus: 'connected' | 'error' | 'unconfigured';
  }

  let { providers }: { providers: Provider[] } = $props();
</script>

<Accordion.Root type="single" class="space-y-2">
  {#each providers as provider (provider.type)}
    <Accordion.Item value={provider.type} class="border rounded-lg">
      <Accordion.Header>
        <Accordion.Trigger class="flex w-full items-center justify-between p-4">
          <div class="flex items-center gap-3">
            <!-- Status indicator dot -->
            {#if provider.connectionStatus === 'connected'}
              <span class="h-2.5 w-2.5 rounded-full bg-green-500"></span>
            {:else if provider.connectionStatus === 'error'}
              <span class="h-2.5 w-2.5 rounded-full bg-red-500"></span>
            {:else}
              <span class="h-2.5 w-2.5 rounded-full bg-gray-300"></span>
            {/if}
            <span class="font-medium capitalize">{provider.type}</span>
          </div>
          <ChevronDown class="h-4 w-4 transition-transform data-[state=open]:rotate-180" />
        </Accordion.Trigger>
      </Accordion.Header>
      <Accordion.Content class="p-4 pt-0">
        <ProviderForm provider={provider} />
      </Accordion.Content>
    </Accordion.Item>
  {/each}
</Accordion.Root>
```

### Pattern 2: Masked API Key Input
**What:** Input showing `****...abc1` format with test button
**When to use:** API key entry fields
**Example:**
```svelte
<!-- Source: CONTEXT.md user decisions -->
<script lang="ts">
  let apiKey = $state('');
  let maskedDisplay = $state('');
  let testing = $state(false);
  let testResult = $state<'success' | 'error' | null>(null);

  function maskKey(key: string): string {
    if (key.length <= 4) return '****';
    return `****...${key.slice(-4)}`;
  }

  function handleInput(e: Event) {
    apiKey = (e.target as HTMLInputElement).value;
    maskedDisplay = maskKey(apiKey);
  }

  async function testConnection() {
    testing = true;
    try {
      const result = await settingsApi.testProvider(providerType, apiKey);
      testResult = result.success ? 'success' : 'error';
    } finally {
      testing = false;
    }
  }
</script>

<div class="space-y-2">
  <div class="relative">
    <input
      type="password"
      value={apiKey}
      oninput={handleInput}
      placeholder="Enter API key"
      class="w-full rounded-md border px-3 py-2 pr-24"
    />
    {#if apiKey}
      <span class="absolute right-12 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
        {maskedDisplay}
      </span>
    {/if}
  </div>
  {#if testResult === 'error'}
    <p class="text-sm text-destructive">Invalid or expired API key</p>
  {/if}
</div>
```

### Pattern 3: Memory Slider with Mode Toggle
**What:** Slider with % vs GB toggle, showing current value
**When to use:** Model pool memory configuration
**Example:**
```svelte
<!-- Source: bits-ui Slider docs + CONTEXT.md -->
<script lang="ts">
  import { Slider, Switch } from 'bits-ui';

  let usePercentMode = $state(true);
  let memoryValue = $state(80); // % or GB depending on mode
  let totalMemoryGb = $state(32); // From system info

  const displayValue = $derived(
    usePercentMode ? `${memoryValue}%` : `${memoryValue} GB`
  );

  const max = $derived(usePercentMode ? 100 : totalMemoryGb);
  const step = $derived(usePercentMode ? 5 : 1);
</script>

<div class="space-y-4">
  <div class="flex items-center justify-between">
    <label class="text-sm font-medium">Memory Limit</label>
    <div class="flex items-center gap-2">
      <span class="text-sm">%</span>
      <Switch.Root bind:checked={usePercentMode}>
        <Switch.Thumb />
      </Switch.Root>
      <span class="text-sm">GB</span>
    </div>
  </div>

  <Slider.Root
    type="single"
    bind:value={memoryValue}
    min={usePercentMode ? 10 : 1}
    {max}
    {step}
    class="relative flex w-full items-center"
  >
    <div class="relative h-2 w-full rounded-full bg-gray-200">
      <Slider.Range class="absolute h-full rounded-full bg-mlx-500" />
    </div>
    <Slider.Thumb index={0} class="h-5 w-5 rounded-full border-2 bg-white shadow" />
  </Slider.Root>

  <div class="text-center text-lg font-semibold">{displayValue}</div>
</div>
```

### Pattern 4: Drag-Drop Rule Cards
**What:** Sortable cards for routing rules with priority by position
**When to use:** Model routing rule configuration
**Example:**
```svelte
<!-- Source: @rodrigodagostino/svelte-sortable-list docs -->
<script lang="ts">
  import { SortableList, sortItems } from '@rodrigodagostino/svelte-sortable-list';
  import type { BackendMapping } from '$api';

  let rules = $state<BackendMapping[]>([]);

  function handleDragEnd(e: { draggedItemIndex: number; targetItemIndex?: number; isCanceled: boolean }) {
    const { draggedItemIndex, targetItemIndex, isCanceled } = e;
    if (!isCanceled && targetItemIndex !== undefined && draggedItemIndex !== targetItemIndex) {
      rules = sortItems(rules, draggedItemIndex, targetItemIndex);
      // Update priorities based on new positions
      updatePriorities();
    }
  }

  async function updatePriorities() {
    const updates = rules.map((rule, index) => ({
      id: rule.id,
      priority: rules.length - index // Top = highest priority
    }));
    await settingsApi.updateRulePriorities(updates);
  }
</script>

<SortableList.Root ondragend={handleDragEnd} gap={8}>
  {#each rules as rule, index (rule.id)}
    <SortableList.Item id={rule.id.toString()} {index}>
      <RuleCard {rule} onDelete={() => deleteRule(rule.id)} />
    </SortableList.Item>
  {/each}
</SortableList.Root>
```

### Pattern 5: Fernet API Key Encryption (Backend)
**What:** Symmetric encryption for storing API keys
**When to use:** Before persisting cloud credentials to database
**Example:**
```python
# Source: cryptography.io Fernet docs
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from mlx_manager.config import settings

# Derive Fernet key from jwt_secret + app salt
# Salt stored in ~/.mlx-manager/.encryption_salt
SALT_FILE = settings.database_path.parent / ".encryption_salt"

def _get_or_create_salt() -> bytes:
    """Get or create persistent salt for key derivation."""
    if SALT_FILE.exists():
        return SALT_FILE.read_bytes()
    salt = os.urandom(16)
    SALT_FILE.write_bytes(salt)
    return salt

def _get_fernet() -> Fernet:
    """Get Fernet instance with derived key."""
    salt = _get_or_create_salt()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=1_200_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(settings.jwt_secret.encode()))
    return Fernet(key)

def encrypt_api_key(plain_key: str) -> str:
    """Encrypt API key for storage."""
    f = _get_fernet()
    return f.encrypt(plain_key.encode()).decode()

def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt API key from storage."""
    f = _get_fernet()
    return f.decrypt(encrypted_key.encode()).decode()
```

### Anti-Patterns to Avoid
- **Storing plain text API keys:** Always encrypt before database storage
- **Exposing encrypted keys in API responses:** Never return `encrypted_api_key` field
- **Reconstructing Fernet on every call:** Cache the Fernet instance per request lifecycle
- **Mixing config and runtime data:** Keep settings page for configuration only; runtime metrics belong on server tiles

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Collapsible sections | Custom show/hide divs | bits-ui Accordion | Accessibility, animation, state management |
| Range slider | HTML range input | bits-ui Slider | Touch support, thumb styling, ticks |
| Drag-drop list | Custom drag events | svelte-sortable-list | Keyboard nav, touch, a11y announcements |
| API key encryption | Base64 or custom cipher | Fernet | Authenticated encryption, tamper detection |
| Toggle switch | Checkbox with CSS | bits-ui Switch | Focus management, a11y, data attributes |
| Searchable dropdown | Custom input + list | bits-ui Combobox | Already proven pattern in ProfileSelector |

**Key insight:** UI configuration involves many edge cases (keyboard navigation, touch drag, focus trapping). bits-ui and svelte-sortable-list handle these correctly.

## Common Pitfalls

### Pitfall 1: Losing Encryption Key
**What goes wrong:** API keys become permanently unreadable
**Why it happens:** `jwt_secret` changes or salt file deleted
**How to avoid:** Document that changing `jwt_secret` invalidates stored API keys; salt file in data dir is critical
**Warning signs:** `InvalidToken` exceptions on decrypt

### Pitfall 2: Drag-Drop State Desync
**What goes wrong:** UI shows different order than database
**Why it happens:** Optimistic updates without backend confirmation
**How to avoid:** Confirm backend save before updating local state; show loading indicator during reorder
**Warning signs:** Refresh shows different order than before

### Pitfall 3: Memory Slider Edge Cases
**What goes wrong:** Users can set 0% or values exceeding system memory
**Why it happens:** No bounds validation
**How to avoid:** Set min/max dynamically based on system memory; validate on backend
**Warning signs:** Server fails to start with invalid memory settings

### Pitfall 4: Accordion State Not Persisted
**What goes wrong:** Accordion resets to collapsed on page navigation
**Why it happens:** State only in component
**How to avoid:** Use URL state or localStorage to remember open sections
**Warning signs:** User frustration from having to re-expand sections

### Pitfall 5: Connection Test Race Condition
**What goes wrong:** Old test result displays after user changes key
**Why it happens:** Multiple concurrent API calls
**How to avoid:** Track request ID or abort controller; clear result on input change
**Warning signs:** "Connected" status after entering invalid key

### Pitfall 6: Warning Badge Not Updating
**What goes wrong:** Rule shows warning after provider configured
**Why it happens:** Rule list not refetching after provider save
**How to avoid:** Invalidate/refetch routing rules when provider status changes
**Warning signs:** Manual page refresh needed to clear warnings

## Code Examples

Verified patterns from official sources:

### Settings Route Page Structure
```svelte
<!-- Source: Project patterns, CONTEXT.md decisions -->
<script lang="ts">
  import { ProviderSection, ModelPoolSettings, RoutingRulesSection } from '$components/settings';
  import { settingsStore } from '$stores';
  import { onMount } from 'svelte';

  onMount(() => {
    settingsStore.load();
  });
</script>

<div class="max-w-4xl mx-auto p-6 space-y-8">
  <h1 class="text-2xl font-bold">Settings</h1>

  <!-- Providers Section -->
  <section>
    <h2 class="text-lg font-semibold mb-4">Cloud Providers</h2>
    <ProviderSection />
  </section>

  <!-- Model Pool Section -->
  <section>
    <h2 class="text-lg font-semibold mb-4">Model Pool</h2>
    <ModelPoolSettings />
  </section>

  <!-- Routing Rules Section -->
  <section>
    <h2 class="text-lg font-semibold mb-4">Model Routing Rules</h2>
    <RoutingRulesSection />
  </section>
</div>
```

### Settings API Endpoints
```python
# Source: Project patterns, Phase 10 schemas
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_manager.database import get_db
from mlx_manager.models import (
    BackendMapping, BackendMappingCreate, BackendMappingResponse,
    CloudCredential, CloudCredentialCreate, CloudCredentialResponse,
)
from mlx_manager.services.encryption_service import encrypt_api_key, decrypt_api_key

router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("/providers", response_model=list[CloudCredentialResponse])
async def list_providers(session: AsyncSession = Depends(get_db)):
    """List configured providers (without API keys)."""
    result = await session.execute(select(CloudCredential))
    return list(result.scalars().all())

@router.post("/providers", response_model=CloudCredentialResponse)
async def create_or_update_provider(
    data: CloudCredentialCreate,
    session: AsyncSession = Depends(get_db),
):
    """Create or update provider credentials."""
    # Check for existing
    result = await session.execute(
        select(CloudCredential).where(CloudCredential.backend_type == data.backend_type)
    )
    credential = result.scalar_one_or_none()

    encrypted_key = encrypt_api_key(data.api_key)

    if credential:
        credential.encrypted_api_key = encrypted_key
        credential.base_url = data.base_url
        credential.updated_at = datetime.now(tz=UTC)
    else:
        credential = CloudCredential(
            backend_type=data.backend_type,
            encrypted_api_key=encrypted_key,
            base_url=data.base_url,
        )
        session.add(credential)

    await session.commit()
    await session.refresh(credential)
    return credential

@router.post("/providers/{backend_type}/test")
async def test_provider_connection(
    backend_type: str,
    session: AsyncSession = Depends(get_db),
):
    """Test connection to cloud provider."""
    result = await session.execute(
        select(CloudCredential).where(CloudCredential.backend_type == backend_type)
    )
    credential = result.scalar_one_or_none()
    if not credential:
        raise HTTPException(404, "Provider not configured")

    api_key = decrypt_api_key(credential.encrypted_api_key)
    # Test connection logic here...
    return {"success": True}
```

### Rule Card Component
```svelte
<!-- Source: CONTEXT.md decisions -->
<script lang="ts">
  import { Card, Badge, Button, Select } from '$components/ui';
  import { GripVertical, Trash2, AlertTriangle } from 'lucide-svelte';
  import type { BackendMapping } from '$api';

  interface Props {
    rule: BackendMapping;
    hasWarning?: boolean; // Unconfigured provider
    onDelete: () => void;
  }

  let { rule, hasWarning = false, onDelete }: Props = $props();

  const patternTypes = [
    { value: 'exact', label: 'Exact' },
    { value: 'prefix', label: 'Prefix' },
    { value: 'regex', label: 'Regex' },
  ];
</script>

<Card class="p-4">
  <div class="flex items-center gap-3">
    <!-- Drag handle -->
    <div class="cursor-grab text-muted-foreground">
      <GripVertical class="h-5 w-5" />
    </div>

    <!-- Pattern info -->
    <div class="flex-1 min-w-0">
      <div class="flex items-center gap-2">
        <Badge variant="secondary">{rule.pattern_type ?? 'exact'}</Badge>
        <code class="text-sm font-mono truncate">{rule.model_pattern}</code>
        {#if hasWarning}
          <Badge variant="warning" class="flex items-center gap-1">
            <AlertTriangle class="h-3 w-3" />
            Unconfigured
          </Badge>
        {/if}
      </div>
      <p class="text-sm text-muted-foreground mt-1">
        Routes to <span class="font-medium capitalize">{rule.backend_type}</span>
        {#if rule.fallback_backend}
          <span class="text-xs">(fallback: {rule.fallback_backend})</span>
        {/if}
      </p>
    </div>

    <!-- Actions -->
    <Button variant="ghost" size="sm" onclick={onDelete}>
      <Trash2 class="h-4 w-4" />
    </Button>
  </div>
</Card>
```

### Model Pool Memory Gauge (Server Tile Addition)
```svelte
<!-- Source: CONTEXT.md - belongs on server tiles, not config page -->
<script lang="ts">
  import { MetricGauge } from '$components/servers';

  interface Props {
    poolUsedMb: number;
    poolMaxMb: number;
  }

  let { poolUsedMb, poolMaxMb }: Props = $props();

  const poolPercent = $derived(
    poolMaxMb > 0 ? (poolUsedMb / poolMaxMb) * 100 : 0
  );
</script>

<!-- Add alongside existing Memory and CPU gauges -->
<MetricGauge
  value={poolPercent}
  label="Pool"
  size="md"
  thresholds={{ warning: 70, danger: 90 }}
/>
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Plain localStorage API keys | Encrypted database storage | Security best practice | Keys protected at rest |
| HTML5 drag events | Library-based sortable | 2024+ | Touch support, accessibility |
| Custom accordion | bits-ui Accordion | bits-ui v1.0 | Proper ARIA, transitions |
| Percentage-only memory | Toggle between % and GB | User feedback | More intuitive for power users |

**Deprecated/outdated:**
- `svelte-dnd-action`: Not fully Svelte 5 compatible; use `svelte-sortable-list` v2.x
- Manual SSE for connection test: Use simple POST with timeout

## Open Questions

Things that couldn't be fully resolved:

1. **Eviction policy storage location**
   - What we know: User selects LRU/LFU/TTL in UI
   - What's unclear: Store as profile setting or global server config?
   - Recommendation: Store as global config in a new `server_config` table

2. **Preload list persistence**
   - What we know: User selects models from downloaded list
   - What's unclear: Store model IDs or full paths?
   - Recommendation: Store model paths for robustness across renames

3. **Connection test implementation**
   - What we know: Need to verify API key works
   - What's unclear: OpenAI/Anthropic endpoints for lightweight validation
   - Recommendation: Use `/v1/models` (OpenAI) or `/v1/messages` with minimal request (Anthropic)

## Sources

### Primary (HIGH confidence)
- [bits-ui Accordion](https://www.bits-ui.com/docs/components/accordion) - Component API and props
- [bits-ui Slider](https://www.bits-ui.com/docs/components/slider) - Slider with ticks and thumbs
- [bits-ui Switch](https://www.bits-ui.com/docs/components/switch) - Toggle switch component
- [cryptography Fernet](https://cryptography.io/en/latest/fernet/) - Encryption API and key rotation
- [@rodrigodagostino/svelte-sortable-list](https://github.com/rodrigodagostino/svelte-sortable-list) - Svelte 5 drag-drop

### Secondary (MEDIUM confidence)
- Phase 10 Research (`10-RESEARCH.md`) - CloudCredential and BackendMapping schemas
- Existing codebase patterns - ProfileSelector.svelte, MetricGauge.svelte, ConfirmDialog.svelte

### Tertiary (LOW confidence)
- WebSearch for connection test endpoints - Needs validation with actual API calls

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - bits-ui already installed, patterns proven in codebase
- Architecture: HIGH - Follows existing project structure exactly
- Pitfalls: HIGH - Based on similar configuration UIs and encryption best practices
- API key encryption: HIGH - Fernet is well-documented, cryptography lib already available

**Research date:** 2026-01-29
**Valid until:** 2026-02-28 (30 days - stable UI libraries, well-documented encryption)
