# ENS 491 — Recursive Hierarchy Tasarım Dokümanı

> Bu doküman PN-3'e (mimari-özgün custom PN) başlamadan önce yazılması gereken kağıt tasarımı.  
> **🔒 Kilitli:** Supervisor onaylı veya tasarım toplantısında netleştirilmiş kararlar.  
> **⚠️ Açık:** Supervisor onayı bekleniyor veya empirik olarak netleşecek.  
> **❓ Belirsiz:** Henüz tartışılmadı, ilerleyen aşamada ele alınacak.

---

## 1. Temel Notasyon

**`{n, m}`** — sistemdeki herhangi bir sütunu tanımlar.

- `n` → katman indeksi. `n=0` en alt seviye (primitive skills). `n` arttıkça soyutlama artar.
- `m` → o katmandaki sütun sırası. `m=1`'den başlar, soldan sağa büyür.

Her iki boyut da **runtime'da büyür** — sistem başlatılırken sabit bir derinlik veya genişlik tanımlanmaz.

```
Katman n=1:   {1,1}         {1,2}         {1,3}  ...
               ↑              ↑              ↑
Katman n=0:  {0,1}  {0,2}  {0,3}  {0,4}  {0,5}  ...
```

---

## 2. Sütunun Çift Rolü

Her `{n, m}` sütunu **aynı anda** iki şeydir:

1. **Standalone policy** — kendi katmanındaki görev için bağımsız bir PPO politikası.
2. **Option** — bir üst katmanın meta-controller'ı tarafından çağrılabilecek macro-action.

Bu çift rol mimarinin temelidir. Bir sütun önce policy olarak eğitilir, stabilize olunca bir üst katmana option olarak sunulur.

---

## 3. Hiyerarşi Yapısı

### 3.1 Neden DAG, ağaç değil ⚠️

Aynı primitive sütun (`{0, m}`) birden fazla üst katman meta-controller'ı tarafından paylaşılabilir. Örneğin "kapıya git" becerisi hem DoorKey hem KeyCorridor görevlerinde kullanılıyorsa aynı `{0, m}` sütununa bağlı iki ayrı meta-controller olabilir.

Bu yapı ağaç değil **Directed Acyclic Graph (DAG)**:
- **Directed:** Bağlantılar her zaman düşük `n`'den yüksek `n`'e (alt katmandan üst katmana).
- **Acyclic:** Öğrenme tek yönlü — yeni sütunlar eski sütunlara bakabilir, tersi mümkün değil. Acyclic özelliği yapısal olarak garanti altında.

> **⚠️ Supervisor onayı bekleniyor.** DAG varsayımı üzerine tasarım yapılıyor ama kesinleştirilmedi.

### 3.2 Meta-controller'ın görüş alanı 🔒

Her katmandaki meta-controller **yalnızca kendi katmanını** görür:
- Mevcut ortam gözlemi
- Kendi katmanındaki mevcut aktif option'ların durumu

Bir üst ya da alt katmanın iç durumuna doğrudan erişimi yoktur.

---

## 4. Lateral Bağlantılar

### 4.1 Tanım 🔒

Lateral bağlantılar **aynı katman içinde, soldan sağa** tanımlıdır:

```
{n, m-1}  →  {n, m}
```

`{n, m}` sütunu eğitilirken `{n, m-1}` sütununun ara katman aktivasyonlarına erişebilir. Bu transfer mekanizmasıdır — yeni sütun eski sütundan öğrenir.

### 4.2 Kapsam kısıtlaması 🔒

Lateral bağlantı yalnızca **aynı meta-controller'ın çocukları arasında** geçerlidir. Farklı meta-controller'lara bağlı sütunlar arasında lateral bağlantı yoktur — cross-MC bilgi transferi hiyerarşi seviyesinde, option mekanizmasıyla gerçekleşir.

### 4.3 Inference sırasında lateral bağlantılar ⚠️

Dondurulan `{n, m-1}` sütunu bir üst katmanın option'ı olarak çağrıldığında lateral bağlantıları aktif kalır mı?

**Mevcut tasarım görüşümüz:** Evet — frozen sütun forward pass yapar, aktivasyonları `{n, m}` tarafından okunur. Donma sadece gradient'ı keser, forward pass'ı değil.

> **⚠️ Supervisor onayı bekleniyor.**

---

## 5. Option Wrapping

### 5.1 Bir sütun ne zaman option olur? 🔒

Eğitim ve stabilizasyon sonrasında. "Stabilize" kriteri:

- Reward curve plateau'ya ulaşmış
- Performans belirli bir eşiğin üzerinde tutarlı

Stabilize olmamış sütun meta-controller'a option olarak sunulmaz.

### 5.2 Option'ın bileşenleri (Sutton et al. 1999)

Her option üç parçadan oluşur:

| Bileşen | İçerik |
|---|---|
| **Initiation set** | Option'ın başlatılabileceği state'ler. Şimdilik tüm state uzayı (her yerden çağrılabilir). ❓ Kısıtlanmalı mı? |
| **Policy** | Sütunun eğitilmiş PPO politikası. Frozen. |
| **Termination condition** | Ne zaman durulacak. ⚠️ Bakınız Bölüm 6. |

---

## 6. Termination ve Recursive Çağrı

### 6.1 Tek seviye çağrı

```
Meta-controller (n=1)  →  option {0, m} çağır
{0, m} çalışır
{0, m} terminates
Meta-controller (n=1) kontrolü geri alır
```

### 6.2 Recursive çağrı ⚠️

```
Meta-controller (n=2)  →  option {1, k} çağır
{1, k} kendi içinde meta-controller gibi davranır
{1, k}  →  option {0, m} çağır
{0, m} terminates → {1, k}'ya sinyal
{1, k} terminates → MC(n=2)'ye sinyal
MC(n=2) kontrolü geri alır
```

Termination **aşağıdan yukarıya propagate eder.** Her seviye kendi termination kararını verir, bir üst seviyeyi bilgilendirir.

**Termination alternatifleri (empirik karşılaştırılacak):**

| Strateji | Mekanizma | Risk |
|---|---|---|
| Sabit adım limiti | Option en fazla K adım çalışır | K'nın göreve göre ayarlanması gerekiyor |
| Öğrenilen termination | Option-Critic tarzı — termination ağı karar verir | Eğitim instabilitesi |
| GRU task-change sinyali | GRU yeni görev algılayınca mevcut option'ı sonlandır | Modüller arası bağımlılık artar |

> **⚠️ Recursive termination propagation'ın tam mekaniği supervisor toplantısında netleştirilecek.**

---

## 7. Runtime Growth: Yeni Sütun mu, Yeni Katman mı?

### 7.1 Yeni sütun açma (mevcut katmanda genişleme)

Tetik: AE reconstruction error threshold'u aşıldı **ve/veya** mevcut policy reward plateau'da.

```
{n, m} → sistem yeni görevi algılar → {n, m+1} sütun açılır → eğitilir → stabilize → option'a eklenir
```

### 7.2 Yeni katman açma (hiyerarşi derinleşiyor)

Tetik: Mevcut görev, mevcut katmandaki primitive option'ların **kombinasyonunu** gerektiriyor.

**Mevcut görüş:** Bu "category detection" problemi — hangi sinyalin yeni katman açacağını belirlemek açık bir araştırma sorusudur. Düşünülen yaklaşımlar:

- Mevcut katmanın meta-controller'ı option'lar arası geçişi çok sık yapıyorsa (yüksek switching) → soyutlama eksik → yeni katman
- Reward decomposition: alt katman ödüllendirilip üst katman ödüllendirilmiyorsa → ara soyutlama gerekiyor
- World model tabanlı karar (Phase 3 / scope dışı şimdilik)

> **⚠️ Supervisor toplantısında tartışılacak.** Şimdilik yeni katman açma kararı manuel / explicit olacak — Phase 1-2 boyunca yeni katman açılmayacak, Phase 3'te ele alınacak.

---

## 8. Inference Akışı — Uçtan Uca

Sistemin bir adımı şöyle işler:

```
1. Ortamdan gözlem al: obs_t

2. AE: reconstruction_error(obs_t)
   - Yüksekse → yeni görev → yeni sütun süreci başlat
   - Düşükse → adım 3'e geç

3. GRU: task_id = classify(obs_{t-N:t})
   - Hangi görevdeyiz?

4. Meta-controller (en üst aktif katman):
   - State: obs_t + task context
   - Çıktı: hangi option çağırılacak

5. Seçilen option çalışır:
   - Kendi içinde recursive olarak aynı akışı yapabilir
   - Termination condition sağlanınca durur

6. Reward, meta-controller'a döner
   - Alt katmandaki reward sinyali ile meta-controller reward sinyali AYRI tutulur
   - Merge edilmez
```

---

## 9. Implementasyon Gereksinimleri (PN-3 için)

Bu tasarımı kodlamak için `Column` sınıfının şu özellikleri taşıması gerekiyor:

```python
class Column:
    n: int                    # katman indeksi
    m: int                    # sütun sırası
    policy: PPOPolicy         # eğitilmiş politika
    frozen: bool              # True ise gradient yok
    lateral_source: Column | None  # {n, m-1} referansı
    is_option: bool           # meta-controller'a sunuldu mu?
    sub_layer: MetaController | None  # None ise leaf (n=0)
```

`sub_layer = None` → primitive column (leaf node)  
`sub_layer = MetaController(...)` → bu sütun hem policy hem üst meta-controller

Bu yapı recursive tanım — sütun kendi içinde bir alt meta-controller barındırabilir. Baştan böyle tasarlanmazsa PN-3 sonrasında rewrite gerekir.

---

## 10. Açık Sorular — Supervisor Toplantısı Gündemi

| # | Soru | Neden kritik |
|---|---|---|
| H-1 | DAG yapısı doğru mu? Aynı primitive sütun birden fazla MC'ye bağlanabilir mi? | Tüm mimari buna göre şekilleniyor |
| H-2 | Recursive MC→MC option çağrısında termination propagation tam olarak nasıl? | PN-3 implementasyonu bunu varsayıyor |
| H-3 | Lateral bağlantılar frozen sütun inference sırasında aktif kalıyor mu? | Forward pass davranışı değişir |
| H-4 | Yeni katman açma sinyali ne olacak? Manuel mi, otomatik mi? | Phase 2 scope'unu belirliyor |
| H-5 | Option initiation set kısıtlanacak mı, yoksa everywhere başlatılabilir mi? | Meta-controller exploration'ını etkiliyor |
