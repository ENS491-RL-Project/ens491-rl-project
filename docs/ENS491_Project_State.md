# ENS 491 — Project State & Roadmap

> Bu doküman projenin teknik hafızası. Ekip toplantılarında, supervisor görüşmelerinde ve implementasyon kararlarında referans nokta bu.
>
> **Ne değişmez:** Kilitlenmiş mimari kararlar — bunlar tartışılmaz.  
> **Ne empirik:** Tasarım alternatifleri — hepsini implement et, sonuçları karşılaştır, kazan.  
> **Ne nerede:** Mevcut durum ve sıradaki adımlar.

---

## Story Points Referansı

| SP | Anlam |
|----|-------|
| 1 | Trivial. Nasıl yapılacağı zaten biliniyor. |
| 2 | Straightforward. Küçük belirsizlik. |
| 3 | Moderate. Birkaç bilinmeyen var. |
| 5 | Significant. Tasarım kararı + implementation birlikte. |
| 8 | Complex. Birden fazla bilinmeyen, iterasyon gerektirir. |
| 13 | Very complex. Muhtemelen alt görevlere bölünmeli. |

---

## Kilitlenmiş Mimari Kararlar

Bunlar tartışmaya kapalı. Değiştirilmesi gerekirse önce supervisor onayı, sonra bu dokümanı güncelle.

### Continual Learning Backbone: Progressive Networks

- Her görev için ayrı sütun (`{n, m}` notasyonu: `n` = katman, `m` = sütun sayısı)
- Sütun eğitimi bitince ağırlıklar **dondurulur** — bir daha değişmez
- Lateral bağlantılar **aynı meta-controller'ın çocukları arasında** tanımlı: `{n, m-1} → {n, m}`
- Sütun sayısı ve katman sayısı **runtime'da büyür** — mimari baştan recursive/unbounded olarak tasarlanır
- Implementation yolu: Doric → custom PN → mimariye özgün custom PN (bu sıra atlanamaz)

### Hierarchy Yapısı

- Hiyerarşi **ağaç değil, DAG** — aynı primitif sütunlar birden fazla meta-controller tarafından paylaşılabilir (supervisor onayı bekleniyor ama tasarım buna göre yapılıyor)
- Her katmanın meta-controller'ı **yalnızca kendi katmanını** görür
- Hiyerarşi **derinliği sabit değil** — yeni katman runtime'da açılabilir
- `{n, m}` notasyonu: `n` arttıkça soyutlama seviyesi artar, `m` o katmandaki sütun sayısı

### Task Detection

- **Reconstruction error** → observation-level novelty sinyali (yeni görev var mı?)
- **Sub-policy reward** → behavioral insufficiency sinyali (mevcut policy yeterli mi?)
- Bu iki sinyal **birbirinden bağımsız** tutulur, merge edilmez
- AE ve GRU pipeline'da sıralı: AE "yeni mi?" sorusunu, GRU "hangisi?" sorusunu cevaplar

### Options Framework

- Her eğitilmiş sütun bir **option** olarak wrap edilir (Sutton et al. 1999)
- Options **stabilize olduktan sonra** meta-controller'a eklenir — yarı eğitimli option eklenmez
- Meta-controller **PPO** ile eğitilir (SB3)
- Yeni option eklenince meta-controller sıfırdan eğitilmez — exploration bonus ile keşfetmesi sağlanır

### Stack

- Python 3.10, PyTorch + CUDA, MiniGrid (Gymnasium), Stable-Baselines3
- `FlatObsWrapper + MlpPolicy` — CnnPolicy MiniGrid'de çalışmıyor (7×7 obs vs 8×8 kernel)
- Birincil test ortamı: MiniGrid (Empty → FourRooms → DoorKey → KeyCorridor)
- Experiment tracking: Weights & Biases

---

## Açık Empirik Sorular

Bunlar için "doğru cevap" önceden belli değil. Her alternatifi implement et, karşılaştır.

### AE Mimarisi
`Undercomplete AE` → `Sparse AE` → `VQ-VAE`  
Kriter: task boundary'de reconstruction error ne kadar keskin sıçrıyor?  
Referans: Meyer et al. (2024) — aynı karşılaştırmayı MiniGrid'e replicate et.

### AE Threshold
`Sabit global` → `Per-task adaptive` → `Statistical test (KS / Wasserstein)`  
Referans: Dick et al. 2024 (SWOKS) MiniGrid + PPO'da bunu yapıyor.

### AE Güncelleme Protokolü
`Tek global AE, fine-tune` → `Per-task ayrı AE` → `Frozen shared encoder + task-specific head`  
Ayrıca baseline: AE hiç güncellenmiyor — ne kadar bozuluyor?

### GRU Sequence Uzunluğu
`N=5` → `N=20` → `N=50`  
Kriter: classification accuracy vs task-switching latency trade-off.

### GRU Training Paradigması
`Supervised (label'lı)` → `Contrastive (SimCLR tarzı)` → `Clustering (DBSCAN)`  
Supervised zorunlu başlangıç noktası — label-free geçiş araştırma sorusu.

### Meta-Controller Sparse Reward
`Sub-goal completion bonus` → `Potential-based shaping` → `Intrinsic curiosity`  
Kriter: convergence hızı ve final performance.

### Termination Condition
`Sabit adım limiti` → `Öğrenilen termination (Option-Critic tarzı)` → `GRU task-change sinyali`

### Yeni Sütun Açma Sinyali
`Sadece AE threshold` → `Sadece reward plateau` → `AND` → `OR`  
Her kombinasyonu aynı görev sırasında çalıştır, gereksiz sütun açılma oranını ölç.

### Görev Sırası
`Easy→Hard` → `Hard→Easy` → `Benzer önce` → `İzole önce`  
Kriter: BWT, FWT, Average Performance. Sıranın etkisi başlı başına bir bulgu.

---

## Mevcut Durum

### ✅ Tamamlananlar

| Görev | Not |
|---|---|
| PPO baseline — Empty-8x8 | FlatObsWrapper + MlpPolicy, reward ~0.04 → ~0.91 |
| Catastrophic forgetting gösterimi | Fine-tune → Empty'ye dön, performans düşüşü sayısal olarak görüldü |
| Geliştirme ortamı | Windows + RTX 3070 Ti, CUDA 12.1, Miniconda ens491 env, SB3, MiniGrid |
| GitHub sandbox reposu | `ens491-sandbox` aktif |
| `doric_test.py` | Doric kütüphanesi ile tek sütunlu PN, lateral bağlantı mekanizması incelendi |
| Proposal + literature review | 120+ paper, 7 kategori, Zotero entegrasyonu |
| Takım onboarding dokümanı | Hizalama dokümanı tamamlandı |

### 🔄 Devam Eden

| Görev | Durum |
|---|---|
| Progressive Networks implementasyonu | Doric aşamasında. `custom_pn_test.py` henüz yazılmadı |
| Recursive hierarchy tasarımı | Kağıt üzerinde kararlar alındı, koda yansımadı |
| DAG yapısı supervisor onayı | Bekleniyor |

---

## Roadmap

### Phase 1 — Continual Learning Core

**Amaç:** PN bu stack'te çalışıyor mu? AE ve GRU bağımsız çalışıyor mu? Forgetting önleniyor mu?  
Explicit task ID var. Bu phase çalışmadan Phase 2'ye geçmek anlamsız.

#### Progressive Networks

| # | Görev | SP | Durum |
|---|---|---|---|
| PN-1 | `custom_pn_test.py` — sadece PyTorch, iki sütun, bir lateral bağlantı, Empty üzerinde | 5 | ⬜ Sıradaki |
| PN-2 | Doric vs custom PN karşılaştırması — hangisi recursive tasarıma daha doğal fit ediyor? | 3 | ⬜ |
| PN-3 | Mimari-özgün custom PN — `{n,m}` notasyonu, runtime column growth, PPO + SB3 entegrasyonu | 13 | ⬜ |
| PN-4 | İkinci sütun + FourRooms, lateral bağlantı aktivasyonunu görselleştir | 5 | ⬜ |
| PN-5 | Forgetting testi: N görev sonra birinci göreve dön, performans ölç | 3 | ⬜ |
| PN-6 | EWC baseline — aynı görev sırasında, paper için alt sınır | 5 | ⬜ |
| PN-7 | DoorKey ile üçüncü sütun — N görevde scale ediyor mu? | 3 | ⬜ |

> **PN-3 öncesinde:** Recursive hierarchy tasarımını kod yazmadan önce kağıt üzerinde netleştir. DAG yapısı, meta-controller-to-meta-controller option çağrısı, termination propagation. Bu bir milestone — atlamak ileride rewrite anlamına gelir.

#### Autoencoder

| # | Görev | SP | Durum |
|---|---|---|---|
| AE-1 | Convolutional AE baseline — MiniGrid obs (7×7×3), encode/decode pipeline | 3 | ⬜ |
| AE-2 | Threshold analizi — tanıdık vs yabancı görev error dağılımı, overlap ölç | 5 | ⬜ |
| AE-3 | Mimari karşılaştırma: Undercomplete → Sparse AE → VQ-VAE (empirical) | 8 | ⬜ |
| AE-4 | Güncelleme protokolü karşılaştırması: global fine-tune / per-task / frozen encoder (empirical) | 8 | ⬜ |

#### GRU

| # | Görev | SP | Durum |
|---|---|---|---|
| GRU-1 | Veri toplama pipeline — farklı task'lardan label'lı episode dizileri | 2 | ⬜ |
| GRU-2 | Supervised baseline — N=5 vs N=20 vs N=50 karşılaştır (empirical) | 5 | ⬜ |
| GRU-3 | Label-free geçiş denemesi: contrastive / clustering (araştırma sorusu, negatif sonuç da değerli) | 8 | ⬜ |

#### Phase 1 Integration

| # | Görev | SP | Durum |
|---|---|---|---|
| INT-1 | AE + GRU handoff — AE "yeni" deyince GRU devreye giriyor mu? False positive/negative ölç | 5 | ⬜ |

---

### Phase 2 — Hierarchical Control

**Amaç:** Explicit task ID olmadan sistem görevi anlayabiliyor mu? Öğrendiği becerileri macro-action olarak kullanabiliyor mu?  
Bu phase paper'ın asıl katkısı. Phase 2 tamamlanmış sistem publishable.

> Phase 2'ye başlamadan: recursive hierarchy kağıt tasarımı onaylanmış ve Phase 1 PN/AE/GRU çalışıyor olmalı.

#### Dynamic Options & Meta-Controller

| # | Görev | SP | Durum |
|---|---|---|---|
| OPT-1 | PoC: PN olmadan 2 frozen PPO + meta-controller — options framework çalışıyor mu? | 8 | ⬜ |
| OPT-2 | PN sütunlarını option olarak wrap et — frozen column = option, termination test | 5 | ⬜ |
| OPT-3 | Sparse reward stratejisi karşılaştırması: sub-goal bonus / potential-based / curiosity (empirical) | 8 | ⬜ |
| OPT-4 | Yeni option eklenince keşif: exploration bonus vs optimistic initialization (empirical) | 5 | ⬜ |
| OPT-5 | Termination condition karşılaştırması: sabit limit / öğrenilen / GRU sinyali (empirical) | 8 | ⬜ |

#### Phase 2 Integration & Ablation

| # | Görev | SP | Durum |
|---|---|---|---|
| INT-2 | AE + GRU + PN + meta-controller end-to-end, explicit task ID kaldırıldı | 13 | ⬜ |
| INT-3 | Ablation: AE kapalı | 3 | ⬜ |
| INT-4 | Ablation: GRU kapalı | 3 | ⬜ |
| INT-5 | Ablation: lateral bağlantılar kapalı | 3 | ⬜ |
| INT-6 | Görev sırası karşılaştırması: 4 farklı sıra, aynı sistem (empirical) | 8 | ⬜ |

---

### Phase 3 — Planning (Stretch Goal)

**Amaç:** Meta-controller reactive seçim yapmak yerine option graph üzerinde sequence planlayabiliyor mu?  
Phase 2 stable olmadan başlanmaz. Paper Phase 2 sonuçlarına göre yazılıyor, Phase 3 ek katkı.

| # | Görev | SP | Durum |
|---|---|---|---|
| PLN-1 | Option transition graph inşa et — hangi option'dan sonra hangisi başarılı? | 8 | ⬜ |
| PLN-2 | Lookahead ekle — reactive vs deliberative karşılaştırma | 13 | ⬜ |
| PLN-3 | Multi-level hierarchy end-to-end: Level-2 meta-controller, iki seviye çalışıyor mu? | 13 | ⬜ |

---

### Phase 4 — Evaluation & Paper (Phase 2 ile Paralel)

| # | Görev | SP | Durum |
|---|---|---|---|
| EVAL-1 | W&B dashboard: reconstruction error, option frekansı, per-task reward canlı | 3 | ⬜ |
| EVAL-2 | BWT / FWT / Average Performance utility — W&B'ye log'layan | 3 | ⬜ |
| EVAL-3 | Streamlit minimal GUI: aktif sütun, option history, policy görselleştirme | 8 | ⬜ |
| EVAL-4 | Full evaluation: 3+ görev sırası, tüm metrikler, baseline karşılaştırması | 8 | ⬜ |
| EVAL-5 | Paper draft — Phase 2 sonuçlarıyla | 13 | ⬜ |

---

## Kritik Path

Bu sıra bloklanırsa sistem bir araya gelemiyor:

```
PN-1 → PN-2 → PN-3* → PN-4 → OPT-1 → OPT-2 → INT-2
                ↑
        Recursive hierarchy tasarım milestone (kağıt üzerinde)
```

Paralel başlanabilecekler (kritik path'i bloklamıyor):  
`AE-1, AE-2, GRU-1, GRU-2, EVAL-1, EVAL-2`

---

## Supervisor'a Onaylatılacak Açık Sorular

- [ ] DAG yapısı doğru mu? Aynı primitif sütunlar birden fazla meta-controller tarafından paylaşılabilir mi?
- [ ] Meta-controller-to-meta-controller recursive option çağrısı ve termination propagation tam olarak nasıl çalışmalı?
- [ ] Yeni görev yeni katman mı, yeni sütun mu açıyor? Bu kararı ne belirliyor?
- [ ] GRU continual learning: EWC mi, accumulated data üzerinde retraining mi?
- [ ] MiniGrid görev sırası onayı: Empty → FourRooms → DoorKey → KeyCorridor → custom multi-key
- [ ] OPT-1 PoC yaklaşımı onaylandı mı? (frozen PPO policies + meta-controller, PN yok)
