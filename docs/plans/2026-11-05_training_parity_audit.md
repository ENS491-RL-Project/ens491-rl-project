# Eğitim eşitliği incelemesi — 2026-11-05

## Sonuç

18 eğitim koşusu yeniden çalıştırıldı: unlock_door goal-only ve solve_doorkey goal-only/strict, her biri seed 42–44, eski/yeni eşleştirilmiş. Bütün çiftlerde son ağırlıkların, her gerçek PPO güncellemesinin giriş buffer'larının ve güncelleme öncesi/sonrası ağırlıklarının SHA-256 değerleri eşit. Bu sonuç yalnızca bu checkpoint'ler, ayarlar ve seed'ler için geçerli.

| Görev/kol | Seed | Eski env adımı | Yeni env adımı | Süresi değişen karar | Gerçek PPO train çağrısı | Rapordaki update |
|---|---:|---:|---:|---:|---:|---:|
| unlock_door goal | 42 | 34,877 | 34,733 | 3 | 2 | 3 |
| unlock_door goal | 43 | 32,869 | 32,629 | 5 | 2 | 3 |
| unlock_door goal | 44 | 33,369 | 33,129 | 5 | 2 | 3 |
| solve_doorkey goal | 42 | 55,659 | 55,659 | 0 | 2 | 3 |
| solve_doorkey goal | 43 | 48,887 | 48,887 | 0 | 2 | 3 |
| solve_doorkey goal | 44 | 53,930 | 53,930 | 0 | 2 | 3 |
| solve_doorkey strict | 42 | 7,280 | 7,280 | 0 | 1 | 2 |
| solve_doorkey strict | 43 | 7,253 | 7,253 | 0 | 1 | 2 |
| solve_doorkey strict | 44 | 7,240 | 7,240 | 0 | 1 | 2 |

### unlock_door: neden aynı?

Eski kodda 3/5/5 karar, ebeveynin 240 adımlık timeout'unu aşıp 288'e ulaşıyor. Yeni kod her birini 48 adım önce kesiyor. Bu kuyrukların tamamında action=2 (ileri), gözlem hash'i her adımda aynı. PPO'ya verilen başlangıç/bitiş gözlemleri, action, mask, reward ve terminated/truncated bayrakları bütün kararlarda aynı. Yalnızca k ve option_steps değişiyor.

Mevcut MaskablePPO, karar başına sabit gamma=0.99 kullanıyor; info içindeki k öğrenme buffer'ına girmiyor. Timeout bootstrap'ında kullanılan terminal gözlemi de aynı. Dolayısıyla gerçek eğitim girdilerinin aynı olması, ağırlıkların eşit kalmasını açıklıyor. Bunu yalnızca son checkpoint'e bakarak değil, observations/actions/rewards/episode_starts/values/log_probs/advantages/returns/action_masks buffer hash'lerini her train çağrısı öncesinde karşılaştırarak doğruladık.

### solve_doorkey: neden aynı?

Goal-only kolunda kayıtlı geçişler, süreler ve nested mask çağrıları tamamen aynı. Strict kolunda ise her seed'de 512 üst-seviye kararın nested mask kayıtları değişiyor; üst policy'nin geçişleri ve bütün PPO buffer'ları değişmiyor. Maskenin aktarılması etkisiz/çalışmıyor denemez: maske değişiyor ama bu çocuk checkpoint'leri ve ulaşılan durumlarla üst policy'nin eğitim verisine yansımıyor. İç seçimin ve primitive eylemlerin tamamı bu probe tarafından kaydedilmiyor; ölçüm üst-seviye geçişler ve mask çağrıları düzeyinde.

### Ek bulgu: update sayısı fazla raporlanıyor

src/training/node_trainer.py, policy_updates değerini decisions // (n_steps * n_envs) ile hesaplıyor. Ancak EvaluationCallback._on_step() stabilite sağlanınca False döndürüyor. Kurulu sb3_contrib collect_rollouts, bu sinyal üzerine rollout'u tamamlayıp train() çağırmadan çıkıyor. Son toplanan rollout ağırlıklara uygulanmıyor.

Ölçülen gerçek train() çağrıları unlock/solve goal için 2 (rapor 3), solve strict için 1 (rapor 2). Her train çağrısı burada 4 optimizer epoch içeriyor; _n_updates ile train çağrısı farklı birimler. Sayaçta completed_rollouts, train_calls, optimizer_epochs ve collected_decisions ayrı tutulmalı. Bu raporlama düzeltmesi geçmiş ağırlıkları değiştirmez.

## Yöntem ve yeniden çalıştırma

- Yeni kod: 584b4be (tam SHA provenance.json içinde).
- Eski kod: 917665e commit'indeki src/skills/executor.py ve src/envs/option_env.py modülleri, ayrı bir süreçte yükleniyor; trainer, çocuk checkpoint'leri ve diğer ayarlar aynı tutuluyor.
- CPU, torch threads=2, 8 env, n_steps=64, batch_size=64, eval_every=512, n_eval=50, bütçe 51,200 karar; mevcut stabilite kriteriyle erken durdurma.
- Probe hiçbir checkpoint kaydetmez/freeze etmez. Registry yalnızca okunur. Stopping seed'leri kullanılır; yeni bir final test seti açılmaz.
- Provenance: paket sürümleri ve checkpoint/metadata SHA-256 değerleri provenance.json içinde. JSON koşu kayıtlarında geçişler, buffer hash'leri, gerçek güncellemeler ve rapor bulunur.

Repo kökünden, ens491 ortamındaki Python ile:

```sh
python scripts/audit_training_parity.py --variant old --task unlock --seed 42
python scripts/audit_training_parity.py --variant new --task unlock --seed 42
python scripts/audit_training_parity.py --variant old --task solve --strict --seed 42
python scripts/audit_training_parity.py --variant new --task solve --strict --seed 42
```

Seed 43/44 ve solve goal-only için de çalıştırıldı. Bütün dosyalar hazırken:

```sh
python scripts/compare_training_parity.py
```

Güncel test paketi: 267 passed. Yeni MAX_ITERS regresyon testleri dahil.


## Integration

The runtime parity changes are in PR #15. Checkpoint loading and the update counter are a separate change based on main. Old reports without `policy_updates_semantics` retain their original numbers: do not subtract one indiscriminately. The overcount occurs when early stopping discards a rollout on its boundary. New reports declare `completed_train_calls`.
