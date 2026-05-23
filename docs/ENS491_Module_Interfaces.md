# ENS 491 — Modül Interface Spesifikasyonu

> Her modülün input/output kontratı. Paralel geliştirmede bu doküman referans.  
> Bir modülün içi değişebilir — ama bu kontrattan sapılmaz.  
> Değişiklik gerekirse önce burası güncellenir, sonra kod.

---

## Genel Tipler

```python
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

# MiniGrid observation: 7×7×3 uint8 array (FlatObsWrapper sonrası 147-dim float vector)
Observation = torch.Tensor       # shape: (147,) — flattened
ObsSequence  = torch.Tensor       # shape: (N, 147) — son N gözlem
Action       = int                # discrete: 0-6 (MiniGrid action space)
TaskID       = int                # 0-indexed, -1 = bilinmiyor
LayerIdx     = int                # n — hiyerarşi katmanı
ColumnIdx    = int                # m — katman içi sütun sırası
```

---

## Modül 1: Autoencoder (AE)

**Sorumluluğu:** "Bu gözlemi daha önce gördüm mü?"

```python
@dataclass
class AEOutput:
    reconstruction_error: float       # MSE — düşük = tanıdık, yüksek = yabancı
    is_novel: bool                    # error > threshold ise True
    latent_z: torch.Tensor            # shape: (latent_dim,) — downstream kullanım için

class AutoencoderModule:
    def encode(self, obs: Observation) -> torch.Tensor:
        """Gözlemi latent space'e sıkıştır."""
        ...

    def decode(self, z: torch.Tensor) -> Observation:
        """Latent vektörden gözlemi yeniden oluştur."""
        ...

    def forward(self, obs: Observation) -> AEOutput:
        """Ana interface. Her adımda çağrılan metot."""
        ...

    def update_threshold(self, recent_errors: list[float]) -> None:
        """Threshold güncelleme — adaptive stratejiler için."""
        ...
```

**Notlar:**
- `is_novel = True` → GRU'ya geçme, yeni sütun sürecini başlat
- `is_novel = False` → GRU devreye girer
- `latent_z` şimdilik kullanılmayabilir ama downstream (GRU input, task representation) için hazır tutulur
- Threshold stratejisi (sabit / adaptive / istatistiksel) AE içinde kapsüllenir — dışarıdan sadece `is_novel` görünür

---

## Modül 2: GRU Task Identifier

**Sorumluluğu:** "Bu hangi görev?"

```python
@dataclass
class GRUOutput:
    task_id: TaskID                   # tahmin edilen görev ID'si (-1 = belirsiz)
    confidence: float                 # [0.0, 1.0]
    all_probs: torch.Tensor           # shape: (num_known_tasks,) — softmax çıktısı

class GRUTaskIdentifier:
    def forward(self, obs_sequence: ObsSequence) -> GRUOutput:
        """Son N gözlemi al, görev tahmin et."""
        ...

    def register_new_task(self, task_id: TaskID) -> None:
        """Yeni görev öğrenilince çıktı boyutunu güncelle."""
        ...
```

**Notlar:**
- `obs_sequence` son N adımın gözlemleri — N hyperparameter (5 / 20 / 50 empirik karşılaştırılacak)
- `task_id = -1` → GRU emin değil, sistemin davranışı TBD (⚠️ supervisor onayı)
- AE `is_novel = False` dediğinde GRU çağrılır — AE `is_novel = True` dediğinde GRU çağrılmaz
- Phase 1'de supervised (label'lı), sonraki aşamada label-free geçiş denenecek

---

## Modül 3: Progressive Networks Column

**Sorumluluğu:** "Bu görevi nasıl yaparım?" + lateral transfer sağla

```python
@dataclass
class ColumnOutput:
    action: Action                              # seçilen eylem
    value: float                                # PPO value estimate
    activations: dict[int, torch.Tensor]        # katman_idx → aktivasyon (lateral için)

class Column:
    n: LayerIdx
    m: ColumnIdx
    frozen: bool
    lateral_source: Optional['Column']          # {n, m-1} referansı, None ise ilk sütun
    sub_layer: Optional['MetaController']       # None ise leaf (primitive), varsa recursive

    def forward(self, obs: Observation) -> ColumnOutput:
        """
        İki farklı davranış — sub_layer'a göre ayrışır:

        Leaf (sub_layer=None):
            - Gözlemi işle, doğrudan action üret.
            - lateral_source varsa aktivasyonlarını çek ve entegre et.
            - frozen=True ise gradient hesaplanmaz.

        Non-leaf (sub_layer=MetaController):
            - Bu Column kendi action'ını üretmez.
            - sub_layer.select_option(obs) çağrılır.
            - Seçilen option kendi forward()'ını çalıştırır (recursive).
            - En alttaki leaf'ten gelen action yukarı taşınır.
            - Bu Column'un policy ağırlıkları meta-controller eğitimi için kullanılır,
              doğrudan env action için değil.
        """
        ...

    def freeze(self) -> None:
        """Ağırlıkları dondur. Eğitim tamamlanınca çağrılır."""
        ...

    def get_activations(self) -> dict[int, torch.Tensor]:
        """Son forward pass'in ara katman aktivasyonları. Lateral için."""
        ...

    def as_option(self) -> 'Option':
        """Bu sütunu Option olarak wrap et."""
        ...
```

**Notlar:**
- `frozen=True` → gradient yok, ama forward pass çalışır (lateral bağlantılar için gerekli) 🔒
- `activations` dict'i `get_activations()` ile bir sonraki sütun tarafından okunur
- `sub_layer=None` → leaf node (n=0, primitive skill)
- `sub_layer=MetaController(...)` → bu sütun hem policy hem bir alt hiyerarşiyi yönetiyor

---

## Modül 4: Option Wrapper

**Sorumluluğu:** Sütunu Sutton et al. (1999) option formatına çevir

```python
@dataclass
class OptionStepOutput:
    action: Action
    terminated: bool        # bu adımda option bitti mi?
    info: dict              # debug bilgisi — adım sayısı, internal reward vs.

class Option:
    column: Column
    option_id: int
    step_count: int         # kaç adımdır çalışıyor

    def step(self, obs: Observation) -> OptionStepOutput:
        """Bir adım çalıştır, termination kontrol et."""
        ...

    def reset(self) -> None:
        """Yeni çağrı başlarken state'i sıfırla."""
        ...

    def can_initiate(self, obs: Observation) -> bool:
        """
        Initiation set kontrolü.
        Şimdilik her state'den başlatılabilir → her zaman True döner.
        ⚠️ İleride kısıtlanabilir.
        """
        return True
```

**Notlar:**
- `terminated=True` → meta-controller kontrolü geri alır
- Termination stratejisi (sabit limit / öğrenilen / GRU sinyali) bu sınıf içinde kapsüllenir
- `step_count` sabit limit stratejisi için kullanılır

---

## Modül 5: Meta-Controller

**Sorumluluğu:** "Hangi option'ı ne zaman çağırmalıyım?"

```python
@dataclass
class MetaControllerOutput:
    selected_option_id: int           # hangi option seçildi
    option: Option                    # seçilen option referansı

class MetaController:
    n: LayerIdx                       # bu meta-controller hangi katmanı yönetiyor
    available_options: list[Option]   # mevcut option listesi

    def select_option(self, obs: Observation) -> MetaControllerOutput:
        """PPO policy ile option seç."""
        ...

    def add_option(self, option: Option) -> None:
        """
        Yeni sütun stabilize olunca çağrılır.
        Exploration bonus ile yeni option'ın keşfedilmesi sağlanır.
        """
        ...

    def step(self, obs: Observation, reward: float, done: bool) -> None:
        """PPO güncelleme adımı."""
        ...
```

**Notlar:**
- Meta-controller'ın reward sinyali alt katman reward'larından **ayrı** tutulur — merge edilmez 🔒
- `add_option()` sıfırdan retraining tetiklemez, mevcut policy üzerine exploration bonus eklenir
- Meta-controller kendisi de bir option olarak wrap edilebilir (recursive yapı için)

---

## Modül 6: Task Lifecycle Manager

**Sorumluluğu:** Modüller arası orkestrasyon — "hangi modül ne zaman devreye giriyor?"

```python
@dataclass
class SystemState:
    current_task_id: TaskID
    active_column: Optional[Column]
    active_option: Optional[Option]
    is_training: bool

class TaskLifecycleManager:
    ae: AutoencoderModule
    gru: GRUTaskIdentifier
    columns: dict[tuple[LayerIdx, ColumnIdx], Column]
    meta_controllers: dict[LayerIdx, MetaController]

    def step(self, obs: Observation) -> Action:
        """
        Her env adımında çağrılır.
        AE → GRU → meta-controller → option → action pipeline'ını yönetir.
        """
        ...

    def on_novel_task_detected(self) -> None:
        """AE is_novel=True döndürdüğünde. Yeni sütun sürecini başlatır."""
        ...

    def on_column_stabilized(self, column: Column) -> None:
        """Sütun eğitimi tamamlanınca. Option wrap + meta-controller'a ekle."""
        ...

    def get_state(self) -> SystemState:
        """Debug ve GUI için anlık sistem durumu."""
        ...
```

---

## Modüller Arası Sinyal Akışı

### Giriş noktası (her adımda)

```
obs_t
  │
  ▼
[AE]──────────────────────────────────────────┐
  │ is_novel=False                             │ is_novel=True
  ▼                                            ▼
[GRU]                                  on_novel_task_detected()
  │ task_id                                    │
  ▼                                            ▼
[TaskLifecycleManager]                 yeni Column açılır
  │                                    eğitilir → stabilize → option'a eklenir
  ▼
[recursive_step(obs, root_mc)]  ← aşağıya bak
```

### Recursive execution (unbounded)

Unbounded yapı burada — her seviye aynı mantığı tekrarlar:

```
recursive_step(obs, MetaController):
  │
  ▼
MetaController.select_option(obs)
  │ → Option seçildi (wraps Column {n, m})
  ▼
Option.step(obs):
  │
  ├─ Column.sub_layer == None  (LEAF)
  │     │
  │     └─ Column.forward(obs) → action
  │           └─ env.step(action) → reward, next_obs, done
  │                 └─ MetaController.step(reward)  ← bu seviyenin MC'si güncellenir
  │
  └─ Column.sub_layer == MetaController  (NON-LEAF)
        │
        └─ recursive_step(obs, Column.sub_layer)
              │  (aynı akış bir alt seviyede tekrar çalışır)
              ▼
             ... (derinlik sınırsız)
              │
              └─ en alttaki leaf action üretir
                    └─ action yukarı taşınır
                          └─ her seviyenin MC'si kendi reward'ını alır
```

**Kilit nokta:** `TaskLifecycleManager` sadece root `MetaController`'ı çağırır — altında kaç seviye olduğunu bilmez, bilmek zorunda değildir. Her seviye kendi altını yönetir. Yeni bir katman eklemek root tanımını değiştirmez, sadece bir Column'un `sub_layer` field'ı doldurulur.

---

## Önemli Kısıtlamalar

| Kural | Açıklama |
|---|---|
| AE ve GRU sinyalleri ayrı | `reconstruction_error` ile `sub_policy_reward` hiçbir zaman tek threshold'a merge edilmez |
| Frozen = forward pass aktif | `frozen=True` sadece gradient'ı keser, aktivasyon üretimi devam eder |
| Option stabilizasyon zorunlu | Yarı eğitimli sütun meta-controller'a eklenmez |
| Reward sinyalleri ayrı | Alt katman reward'ı ile meta-controller reward'ı ayrı kanalda tutulur |
| Recursive tasarım baştan | `sub_layer` field'ı Phase 1'de `None` olsa bile class'ta var olmalı |
