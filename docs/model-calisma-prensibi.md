# Modellerin Çalışma Prensibi

Bu belge, projede kullanılan altı yapay zeka modelinin ne işe yaradığını, girdilerini ve çıktılarını, birbirleriyle
nasıl haberleştiklerini ve hangi modda hangisinin çalıştığını anlatır. Son bölümler, projenin cover yöntemini YuE2
ekibinin önerdiği yöntemle ve NoizAI'nin [yue.noizai.net](https://yue.noizai.net/) sitesiyle karşılaştırır.

Kurulum ve donanım için [yapay-zeka-model-entegrasyonu.md](yapay-zeka-model-entegrasyonu.md), arayüz ve genel
mimari için [ENTEGRASYON.md](ENTEGRASYON.md) belgesine bakın. Dış kaynaklardan alınan bilgiler 15 Ekim 2026
itibarıyla geçerlidir.

---

## 1. Özet

- Şarkıyı üreten tek model **YuE2**'dir (YuE2-3B + YuE2-VAE).
- Diğer dört model yalnızca **Cover** modunda çalışır. Görevleri, referans şarkıyı YuE2'nin okuyabileceği **metne**
  çevirmektir: ABC nota ve söz.
- **YuE2 ses dinleyemez.** Ses dosyası alan bir girdisi yoktur. Referans şarkıdan YuE2'ye ulaşan her bilgi ABC notası
  ya da söz metni olarak gider.

---

## 2. Modeller

| Model | Boyut | Görev | Girdi | Çıktı | Ortam | Cihaz |
|---|---|---|---|---|---|---|
| **YuE2-3B** | 7.3 GB (BF16, ~3.6 milyar parametre) | Şarkıyı üretir | Stil etiketleri, `[verse]`/`[chorus]` etiketli sözler, isteğe bağlı ABC | ABC nota, ses token'ları, ses latent'leri | `yue_ortam` (Python 3.12, torch 2.10) | CUDA > MPS > CPU |
| **YuE2-VAE** | 0.5 GB | Latent'leri sese çevirir | YuE2-3B'nin latent'leri | 48 kHz stereo ses | `yue_ortam` | YuE2 ile aynı |
| **SheetSage2** | ~0.23 GB (MERT-v2 tabanlı) | Kayıttan nota çıkarır (transkripsiyon) | Referans şarkının karışık sesi | ABC (melodi, akor, tempo, ton, bölümler) ve ölçü/bölüm zamanları | `sheetsage_ortam` (Python 3.11, torch 2.8) | CUDA varsa GPU, yoksa CPU |
| **Demucs** (htdemucs) | 84 MB | Kaynak ayrıştırma | Referans şarkı | Vokal, enstrüman ve eşlik kanalları (ses) | `yue_ortam` | CUDA varsa GPU, yoksa CPU |
| **Whisper large-v3-turbo** | 1.6 GB | Çok dilli söz tanıma: dili bulur, Türkçe dışı sözleri yazıya döker | Demucs'un vokal kanalı | Dil kodu, zaman damgalı kelimeler | `yue_ortam` | CUDA > MPS > CPU |
| **Whisper large-v3-turkish-general** | 3.1 GB | Türkçeye ince ayar yapılmış söz tanıma | Demucs'un vokal kanalı (dil `tr` ise) | Zaman damgalı Türkçe kelimeler | `yue_ortam` | CUDA > MPS > CPU |

"Basit analiz" seçeneğinde librosa da kullanılır: vuruş takibi, Krumhansl profilleriyle ton bulma, şablonla akor
tanıma ve pYIN ile melodi çıkarma. Bunlar eğitilmiş model değil, klasik sinyal işleme algoritmalarıdır.

Arayüzler model yüklemez. Gradio arayüzü (`yue-ui/venv`, torch yok) ve web arayüzü (`frontend/`, yalnızca Python
standart kütüphanesi) aynı model sunucusunu ([yue2_server.py](../server/yue2_server.py), port 7861) çağırır.

---

## 3. Her modelin çalışma prensibi

### 3.1 YuE2-3B ve YuE2-VAE

Sunucu bir şarkıyı dört aşamada üretir:

| Aşama | Model | Ne olur |
|---|---|---|
| 1. Nota planı (ABC) | YuE2-3B (otoregresif) | Tempo, ton, ölçüler, bölümler ve (CoT `full` ise) akorları içeren bir ABC nota yazılır. **Dışarıdan ABC verilirse bu aşama atlanır.** |
| 2. Ses token'ları | YuE2-3B (otoregresif) | Nota, sözler ve stile bakılarak sesin her saniyesi için ~25 token üretilir |
| 3. Sentez | YuE2-3B'nin NAR kısmı | Flow matching ile token'lar sürekli latent'lere çevrilir (32 adım, hızlı sentezde 16) |
| 4. Ses çözme | YuE2-VAE | Latent'ler 48 kHz stereo sese çevrilir; sunucu MP3 olarak da kaydeder |

CoT seçenekleri: `full` akorlu ABC, `melody` akorsuz ABC, `off` nota yazılmaz. Dışarıdan ABC vermek için CoT `full`
ya da `melody` olmalıdır.

**Sözler nasıl kullanılır:** YuE2 söz tanıma yapmaz. Sözler istekte ayrı bir metin alanıdır (`lyrics`) ve Cover
modunda da zorunludur. Sözler yalnızca `[bölüm]` etiketli düz metindir; **zaman damgası içermez.** İstekte zaman, BPM ya
da fonem için bir alan yoktur ([protocol.py:81](../../yue_ortam/lib/python3.12/site-packages/yue2/protocol.py#L81)).
YuE2'nin zamanla ilgili bildiği her şey ABC'den gelir: tempo (`Q:`), ölçüler ve `% bölüm` satırları. ABC'de söz,
sözlerde de nota yoktur. İkisini YuE2 token üretirken kendisi eşleştirir: sözlerdeki
**n'inci `[blok]`, notadaki n'inci söylenen bölüme** oturur ve hecelerin notalara dağılımına model karar verir. Bu
nedenle söz bloklarının sayısı ve sırası, notadaki vokal bölümleriyle tutmalıdır.

#### YuE2-3B ile YuE2-VAE'nin farkı

YuE2-3B şarkıyı "besteleyen" ve "çalan" modeldir; YuE2-VAE ise onun ürettiği sıkıştırılmış temsili duyulabilir sese
çeviren çözücüdür. YuE2-3B doğrudan ses üretmez. Ürettiği şey, sesin her saniyesi için 64 boyutlu 25 vektörden oluşan
sıkıştırılmış bir temsildir (**latent**). VAE bu temsili 48 kHz stereo dalga formuna açar.

| | YuE2-3B | YuE2-VAE |
|---|---|---|
| Görev | Müziğin içeriğine karar verir: nota planı, ses token'ları, latent'ler | Latent'leri gerçek sese çevirir |
| Mimari | Transformer çözücü (`YuE2ForCausalLM`): 28 katman, 2048 gizli boyut, 16 sorgu / 8 KV kafası | Evrişimli (convolutional) kodlayıcı-çözücü; 6 katmanda toplam 1920 kat örnek seyreltme |
| Boyut | 7.3 GB, ~3.6 milyar parametre, BF16 | 0.53 GB, ~130 milyon parametre, FP32 |
| Girdi | Stil, sözler, isteğe bağlı ABC (metin) | 64 boyutlu latent dizisi |
| Çıktı | ABC, ses token'ları, latent'ler | 48 kHz stereo ses |
| Müzikten ne anlar | Melodi, söz, yapı, stil | Hiçbir şey; yalnızca sesi "açar" |
| Aşamalar | 1. nota, 2. token, 3. sentez | 4. ses çözme |
| Süre (Mac M4, 49.5 sn şarkı) | 32 + 91 + 165 = ~288 sn | 12 sn |

```
YuE2-3B (AR)  ─► ABC ─► ses token'ları (~25/sn)
YuE2-3B (NAR) ─► latent: saniyede 25 × 64 sayı
YuE2-VAE      ─► ses: saniyede 48.000 × 2 = 96.000 sayı
```

- **Ortak dil 64 boyutlu latent'tir.** YuE2-3B'nin [yapılandırmasında](../../models/YuE2-3B/config.json) ve VAE
  çözücüsünün [yapılandırmasında](../../models/YuE2-Vae/config.json) `latent_dim: 64` yazar; iki modelin arayüzü budur.
- **Hızlar örtüşür.** VAE'nin seyreltme oranı 1920'dir: 48.000 / 1920 = saniyede 25 latent karesi. YuE2-3B de saniyede
  ~25 ses token'ı üretir.
- **Sıkıştırma ~60 kattır.** Latent saniyede 1.600 sayı, ses saniyede 96.000 sayıdır. YuE2-3B'nin uzun şarkıları
  bağlamına sığdırabilmesi, sesi bu sıkıştırılmış temsilde üretmesi sayesindedir.
- **Ses parça parça çözülür.** VAE uzun şarkıyı 1024 karelik (~41 sn) parçalarla çözer. Parça kenarlarında dikiş izi
  kalmasın diye her parçaya 16 karelik pay eklenir. Bellek bütçesi 12 GB'ın altındaysa parçalar 512 kareye iner.

İki ayrı model olmalarının nedenleri:

- **Görevleri farklıdır.** İçeriğe karar vermek milyarlarca parametre gerektirir. Sesi açmak ise hassasiyet ister; bu
  yüzden VAE FP32'de çalışır.
- **VAE değiştirilebilir.** Aynı latent'ler başka bir VAE ile yeniden çözülebilir. Örneğin benchmark sonuçlarını
  tekrarlamak için `YuE2-Vae-legacy` kullanılır. Kaydedilen `latent.npy` dosyasından, şarkıyı yeniden üretmeden yeni ses
  çıkarılabilir.
- **VAE'nin kodlayıcısı üretimde kullanılmaz.** VAE'de sesi latent'e çeviren bir kodlayıcı da vardır. Ama YuE2-3B'nin
  latent alan bir girdisi yoktur, bu yüzden cover için referans şarkıyı VAE ile kodlayıp modele vermek mümkün değildir.
  Cover'ın notayla (SheetSage2) yapılmasının nedeni budur.

#### VAE nedir

VAE, **Variational Autoencoder**'ın kısaltmasıdır. Türkçede "varyasyonel otokodlayıcı" ya da "değişimsel otokodlayıcı"
denir. Adın iki parçası şunu anlatır:

- **Autoencoder (otokodlayıcı):** İki parçalı bir ağdır. **Kodlayıcı (encoder)** girdiyi küçük bir temsile (latent)
  sıkıştırır, **çözücü (decoder)** bu temsilden girdiyi geri kurar. "Oto" kısmı, modelin kendi girdisini geri üretmeyi
  öğrenmesinden gelir; eğitimde hedef, çıkan sesin giren sese olabildiğince benzemesidir.
- **Variational (varyasyonel):** Kodlayıcı her ses parçası için tek bir nokta değil, bir **olasılık dağılımı** üretir:
  bir ortalama ve bir yayılım (standart sapma). Bu, latent uzayını düzgün ve sürekli kılar. Birbirine yakın latent'ler
  birbirine benzeyen seslere karşılık gelir; uzayda "anlamsız boşluklar" kalmaz. YuE2-3B'nin ürettiği latent'lerin
  VAE'den temiz ses olarak çıkabilmesi bu özelliğe dayanır.

YuE2-VAE'de bu doğrudan görülür ([modeling_vae.py:508](../../models/YuE2-Vae/modeling_vae.py#L508)):

- Kodlayıcının çıktısı 128 boyutludur. Bu 128 boyut ikiye bölünür: 64'ü **ortalama**, 64'ü **yayılım**.
- Yayılım `softplus` ile her zaman pozitif tutulur.
- Varsayılan olarak latent olarak ortalama kullanılır. İstenirse dağılımdan örnek de çekilebilir.
- Çözücü, bu 64 boyutlu latent'i alıp 48 kHz stereo sese çevirir.

YuE2'de şarkı üretirken yalnızca **çözücü** çalışır. Latent'leri kodlayıcı değil YuE2-3B üretir; kodlayıcı VAE'nin
eğitiminde gerekir.

### 3.2 SheetSage2

SheetSage2, YuE2 ekibinin kayıttan lead sheet (melodi + akor) çıkarmak için geliştirdiği modeldir. YuE2'nin eğitiminde
de nota hedeflerini üretmek için kullanılmıştır.

- **Nasıl çalışır:** Sesi mono'ya indirir, uzun şarkıları üst üste binen pencerelere böler ve MERT-v2-FullSong
  kodlayıcısıyla işler. Kodlayıcının üzerindeki çözücüler melodi, akor, vuruş, ton ve bölüm yapısını tahmin eder.
- **Kaynak ayrıştırmaya ihtiyaç duymaz.** Karışık sesi doğrudan dinler. Vokal melodisini ve enstrüman melodisini
  kendisi ayrı ayrı tahmin eder (`melody_vocal.mid`, `melody_instrumental.mid`). Ayrıştırmayı sesi bölerek değil, nota
  düzeyinde yapar.
- **Çıktı:** `score.abc` (`V: Vocal` ve `V: Ins` sesleri, akorlar, `% bölüm` satırları), MIDI dosyaları ve zamanlı
  etiket dosyaları (`*.lab`). `melody_only=True` seçeneği akorları atar; resmi cover tarifi bunu kullanır.
- **Projede:** SheetSage2 torch 2.8, YuE2 torch 2.10 istediği için aynı ortamda çalışamazlar.
  [sheetsage.py](../server/sheetsage.py) her analizde SheetSage2'yi kendi ortamında ayrı bir süreç olarak başlatır
  (bkz. [bölüm 4.3](#43-teknik-iletişim)).

### 3.3 Demucs

Demucs bir kaynak ayrıştırma modelidir: karışık bir kaydı, onu oluşturan kaynaklara geri ayırır. Müzikten (melodi,
akor) bir şey anlamaz; yalnızca hangi sesin hangi kaynaktan geldiğini ayırır.

- **Kanallar:** htdemucs 4 kanal üretir: vokal, davul, bas ve "other". **Gitar ayrı bir kanal değildir;** gitar, piyano,
  saz, synth gibi diğer enstrümanların hepsi "other" kanalında kalır.
- **Projede:** [cover.py:91](../server/cover.py#L91) içindeki `separate`, sesi modelin örnekleme hızına çevirir,
  normalize eder, Demucs'u parçalar halinde (%25 örtüşmeyle) çalıştırır ve dört kanalı üç mono sese indirir:
  - **vokal**
  - **other** (vokalsiz kayıtlarda melodi buradan alınır)
  - **eşlik** (vokal dışındaki her şey)
- **Çıktısı ses olduğu için** YuE2'ye doğrudan bir şey söylemez. Çıktısı ancak başka bir araç onu notaya (pYIN, librosa)
  ya da metne (Whisper) çevirdiğinde işe yarar.

### 3.4 Whisper

Whisper bir otomatik konuşma tanıma (ASR) modelidir. Çoğunlukla konuşma kayıtlarıyla eğitildiği için müzik içindeki
vokalde zayıf kalır. Bu yüzden projede **Whisper'ın girdisi her zaman Demucs'un vokal kanalıdır;** Whisper orijinal
şarkıyı hiç duymaz.

[lyrics.py:139](../server/lyrics.py#L139) içindeki `transcribe` şu adımları izler:

1. Vokal 16 kHz'e çevrilir.
2. Enerjiye bakılarak vokalin duyulduğu bölgeler bulunur. Bölgeler en fazla 28 saniyelik parçalara bölünür (bir Whisper
   penceresine sığsın diye).
3. **Dil seçimi:** Dil elle seçilmediyse turbo model en uzun vokal bölgesinde dili tespit eder. Sonuç `tr` ise turbo
   bellekten atılır ve Türkçe model yüklenir. Diğer dillerde yazıya dökme işini turbo yapar. Dil elle seçildiyse tespit
   atlanır ve doğrudan o dilin modeli kullanılır. Bellekte aynı anda tek bir Whisper durur.
4. Her parça kelime düzeyinde zaman damgasıyla yazıya dökülür. Uydurma metinler (halüsinasyon) elenir.
5. Vokalin sessiz olduğu yere düşen kelimeler, yine Demucs'un vokal kanalına bakılarak atılır (`drop_silent`).
6. Kelimeler prozodiye göre satırlara bölünür; Türkçede `i`→`İ` gibi büyük harf kuralları uygulanır.

**Çıktı yalnızca metin değildir:** her kelimenin başlangıç ve bitiş zamanı da vardır. Projede asıl işe yarayan bu
zamanlardır; sözler bölümlere bu zamanlar sayesinde yerleştirilir. **Zamanlar YuE2'ye gönderilmez:** yerleştirme
bittikten sonra atılır ve sözler forma yalnızca `[bölüm]` etiketli düz metin olarak gelir
([lyrics.py:341](../server/lyrics.py#L341)).

### 3.5 Basit analiz (librosa + pYIN)

SheetSage2'den önce yazılmış yöntemdir ([cover.py:341](../server/cover.py#L341)). Notanın temeli Demucs'un
kanallarıdır:

| Kanal | Ne çıkarılır | Nasıl |
|---|---|---|
| vokal + eşlik | Tempo, ton | Vuruş takibi; kroma + Krumhansl profilleri |
| eşlik | Her ölçünün akoru | Kroma + akor şablonları |
| vokal | Melodi | pYIN, 1/16'lık ızgaraya oturtulur |
| other | Melodi (yalnızca vokalsiz kayıtlarda) | pYIN; melodi vokal aralığına taşınır |

Tempo tahmini bazen 4/3 ya da 2 kat yanılır. Bu yüzden yalnızca bu yöntemde elle BPM girilebilir.

---

## 4. Modeller arası iletişim

### 4.1 Nota çıkarıcı: SheetSage2 (varsayılan)

```
              ┌─► 1. SheetSage2 ─► score.abc + her bölümün başlangıç/bitiş saniyesi ──┐
orijinal ses ─┤                                                                        ├─► zamanla eşleştir
              └─► 2. Demucs ─► vokal ─► 3. Whisper ─► kelimeler + saniyeleri ──────────┘
                     (yalnızca "Sözleri tanı" açıksa)
```

**SheetSage2 ile Demucs hiç doğrudan konuşmaz.** İkisi de orijinal şarkıyı [cover.py](../server/cover.py)'den ayrı
ayrı alır; sonuçları en sonda **zaman** üzerinden buluşur. Modeller paralel değil, sırayla çalışır
([cover.py:442](../server/cover.py#L442)):

1. SheetSage2 notayı ve bölüm zamanlarını çıkarır. Örneğin "chorus 42.1–58.3 sn".
2. "Sözleri tanı" açıksa Demucs çalışır. Bu yolda yalnızca **vokal** kanalı kullanılır.
3. Whisper vokali yazıya döker.
4. [lyrics.py:307](../server/lyrics.py#L307) içindeki `place_in_sections` her söz satırının orta zamanına bakar ve
   satırı o zamanı içeren SheetSage2 bölümüne yerleştirir. Ardından bölüm adlarını düzeltir:
   - SheetSage2'nin söylenen bölüm saydığı (ör. "chorus") ama içinde söz olmayan bölüm `instrumental` olur.
   - Enstrümantal saydığı (ör. "intro") ama içinde 2 kelimeden uzun söz olan bölüm `verse` olur.

   Bunun nedeni YuE2'nin n'inci söz bloğunu n'inci söylenen bölüme oturtmasıdır (bkz. [bölüm 3.1](#31-yue2-3b-ve-yue2-vae)).

Bu yolda **notayı yalnızca SheetSage2 yazar.** Whisper yalnızca `% bölüm` satırlarının adlarını ve forma gelen sözleri
etkiler. SheetSage2'nin `melody_vocal.lab` çıktısı şarkıda vokal olup olmadığını gösterir. Whisper hiçbir şey bulamazsa
ve vokal de yoksa kayıt "enstrümantal" sayılır.

### 4.2 Nota çıkarıcı: Basit analiz

```
orijinal ses ─► Demucs ─┬─► vokal ─┬─► pYIN ─► melodi ─────────────┐
                        │          └─► Whisper ─► sözler + zaman ──┤ ("Sözleri tanı" açıksa)
                        ├─► eşlik ─► kroma ─► akorlar, ton ────────┼─► ABC
                        └─► other ─► pYIN ─► melodi (vokalsizse) ──┘
```

SheetSage2 çalışmaz. "Sözleri tanı" açıksa bölüm sınırları ve adları doğrudan söz satırlarının zamanlarından çıkar.
Kapalıysa sınırlar vokal etkinliğinden, adlar kullanıcının yazdığı `[verse]` etiketlerinden gelir. Vokal kanalı
neredeyse sessizse kayıt enstrümantal sayılır ve Whisper hiç çağrılmaz.

### 4.3 Teknik iletişim

| Model | Nasıl çağrılır | Veri nasıl gider/gelir |
|---|---|---|
| Demucs, Whisper | Model sunucusuyla aynı süreçte, normal fonksiyon çağrısı | Bellekteki numpy/torch dizileri |
| SheetSage2 | [sheetsage_runner.py](../server/sheetsage_runner.py), `sheetsage_ortam`'ın Python'uyla ayrı süreç olarak | **Giden:** ses `input.npy` dosyası; model klasörü, örnekleme hızı ve cihaz komut satırı argümanı. **Gelen:** `score.abc`, `sections.json` (bölüm ve ölçü zamanları), `melody_vocal.lab`. **Hata:** sıfırdan farklı çıkış kodu, neden stderr'de |

SheetSage2'nin süreci kapanınca belleğin tamamı geri verilir. Hemen ardından Demucs, Whisper veya YuE2 belleğe ihtiyaç
duyacağı için bu önemlidir.

### 4.4 GPU paylaşımı ve önbellek

- Şarkı üretilirken gelen yeni bir analiz isteği bekletilmez, reddedilir (409).
- CUDA'da boş bellek 8 GiB'ın altındaysa YuE2 analizden önce bellekten çıkarılır
  ([yue2_server.py:595](../server/yue2_server.py#L595)).
- Analiz sonuçları önbellekte tutulur. Anahtar, dosyanın SHA-256 özeti ve analiz ayarlarıdır. Aynı şarkı aynı ayarlarla
  tekrar analiz edilirse hiçbir model çalışmaz ([yue2_server.py:698](../server/yue2_server.py#L698)).

---

## 5. Cover modunda arayüz akışı

### 5.1 Referans melodisi: Yakın ve Serbest

İki mod aynı modelleri kullanır. Aralarındaki tek fark YuE2'ye ne gönderildiğidir
([yue2_ui.py:33](../source/yue2_ui.py#L33), [yue2_ui.py:419](../source/yue2_ui.py#L419)):

| Mod | YuE2'ye giden nota | CoT | Etkisi |
|---|---|---|---|
| **Yakın** | Akorlu nota, olduğu gibi | `full` | Orijinal armoni korunur |
| **Serbest** | Akor sembolleri silinmiş nota (`V: Vocal` ve `V: Ins` melodileri kalır) | `melody` | Eşlik yeni stile göre kendi akorlarını seçer; YuE2 ekibinin stil değiştirirken önerdiği mod |

Mod değiştirildiğinde mevcut nota yeniden çıkarılmadan dönüştürülür. Nota elle değiştirilmişse dönüştürülmez.

### 5.2 "✨ Sözleri tanı"

Basıldığında analiz `recognize_lyrics=True` ile çalışır: Demucs + Whisper devreye girer, tanınan sözler bölüm
etiketleriyle söz kutusuna yazılır. Kullanıcı yanlış kelimeleri düzeltebilir; bölüm etiketlerinin silinmemesi ve
sırasının değiştirilmemesi gerekir.

### 5.3 "Sözleri tanı" kullanmadan doğrudan "Şarkı üret"

MP3 yüklüyken nota kutusu boşsa, "Şarkı üret" notayı önce kendisi çıkarır, sonra şarkıyı üretir
([yue2_ui.py:764](../source/yue2_ui.py#L764)). Nota çıkarıcı SheetSage2 ise bu analizde **yalnızca SheetSage2**
çalışır; Demucs ve Whisper devreye girmez.

- **Sözler zorunludur.** Söz kutusu boşsa analiz başlamaz. Sözler orijinal sözler de olabilir, tamamen yeni sözler de.
- **Bölüm adları SheetSage2'den gelir.** Kullanıcının `[verse]`/`[chorus]` etiketleri notaya yazılmaz. Söz bloklarının
  sayısı ve sırası notadaki söylenen bölümlerle tutmazsa uyarı çıkar ama üretim yine yapılır; sözler yanlış yerlere
  kayabilir.
- **Kutuda nota varsa yeniden çıkarılmaz.** Yeni dosya yüklemek ya da nota çıkarıcı, başlangıç, süre veya BPM'i
  değiştirmek kutuyu boşaltır.

Önerilen sıra:

1. **"Melodiyi çıkar (önizle)"**ye basın.
2. Zaman çizelgesinde SheetSage2'nin bulduğu bölümleri görün.
3. Sözleri bu sırayla, aynı etiketlerle yazın.
4. "Şarkı üret"e basın. Analiz önbellekte olduğu için SheetSage2 ikinci kez çalışmaz.

---

## 6. Hangi modda hangi model çalışır

| Mod | YuE2-3B + VAE | SheetSage2 | Demucs | Whisper turbo | Whisper TR |
|---|---|---|---|---|---|
| **Text to Song** | ✔ (ABC'yi model kendisi yazar; CoT `off`'ta yazmaz) | – | – | – | – |
| **Cover · SheetSage2** (varsayılan) | ✔ (1. aşama atlanır) | ✔ | – | – | – |
| **Cover · SheetSage2 + Sözleri tanı** | ✔ | ✔ | ✔ (yalnızca vokal için) | ✔ (dil tespiti; Türkçe değilse yazıya döker) | ✔ (dil `tr` ise) |
| **Cover · Basit analiz** | ✔ | – | ✔ | – | – |
| **Cover · Basit analiz + Sözleri tanı** | ✔ | – | ✔ | ✔ (vokal varsa) | ✔ (dil `tr` ise) |
| **Mock** (`--mock`) | Model yok, deneme tonu üretilir | – | – | – | – |

Yakın/Serbest seçimi bu tabloyu değiştirmez (bkz. [bölüm 5.1](#51-referans-melodisi-yakın-ve-serbest)).

**Melodiyi ve notayı varsayılan olarak yalnızca SheetSage2 çıkarır.** Demucs notaya yalnızca "Basit analiz"de katkı
yapar. SheetSage2 yolunda Demucs ve Whisper notaya dokunmaz; söz yazma işini otomatikleştirir.

---

## 7. YuE2 ekibinin önerdiği cover yöntemiyle karşılaştırma

YuE2'nin [model kartı](https://huggingface.co/M-A-P/YuE2-3B) ve
[cover kılavuzu](https://github.com/multimodal-art-projection/YuE/blob/main/docs/covers.md) şu tarifi verir. Aynı
tarif yerel [SheetSage2 README'sinde](../../models/SheetSage2/README.md#L139) de yer alır:

1. Kaynak şarkı SheetSage2'ye `melody_only=True` ile verilir: vokal ve enstrüman melodileri kalır, akorlar atılır.
2. Nota, kaynak dinlenerek elle kontrol edilir (notalar, ölçüler, bölüm sırası); transkripsiyon hataları cover'a geçer.
3. Sözleri kullanıcı verir. Model kartı, sözleri bulmayı ya da Qwen3-ASR veya Gemini API ile yazıya dökmeyi önerir.
   Bölüm etiketleri notayla hizalanmalıdır.
4. YuE2'ye `abc` + `cot="melody"` + hedef `style` + `lyrics` verilir. Ses referansı yoktur.

Tarifte Demucs ya da başka bir kaynak ayrıştırma aracı, Whisper ya da otomatik söz tanıma yoktur. Model kartı ayrıca
`cot="melody"` seçeneğinin notadaki akorları kendisi silmediğini belirtir; projede bunu Serbest modundaki
`strip_chords` yapar.

| | Önerilen yöntem | Bu proje |
|---|---|---|
| Nota | SheetSage2, `melody_only=True` | SheetSage2 tam nota çıkarır. **Serbest** akorları siler (sonuç önerilen yöntemle aynı). **Yakın** akorları bırakır (kılavuzdaki "sabit armoni" seçeneği). |
| Kaynak ayrıştırma | Yok | Demucs; SheetSage2 yolunda yalnızca Whisper'a temiz vokal vermek için |
| Sözler | Elle, ya da Qwen3-ASR / Gemini | Whisper turbo + Türkçe Whisper |
| Sözlerin bölümlere yerleşmesi | Elle | Otomatik (`place_in_sections`) |
| CoT | `melody` önerilir; sabit armoni için `full` | Serbest: `melody`, Yakın: `full` |
| Ses referansı | Yok | Yok |

**Sonuç:** "SheetSage2 + Serbest" seçeneği önerilen yöntemin aynısıdır. Demucs ve Whisper projenin kendi eklemeleridir;
önerilen yöntemde elle yapılan söz yazma işini otomatikleştirir.

---

## 8. NoizAI sitesi (yue.noizai.net)

[yue.noizai.net](https://yue.noizai.net/), M-A-P'ın (YuE2'yi geliştiren ekip) değil **NoizAI**'nin sitesidir ("NOIZ ×
YuE2"). Açık kaynak tarafı [NoizAI/YuE2-Turbo](https://github.com/NoizAI/YuE2-Turbo) deposudur. Bu depo YuE2'yi
değiştirmez, hızlandırır (vLLM, toplu sentez). Sitenin kendi arayüzü ve sunucu kodu depoda yoktur.

Sitedeki "AI cover studio" bölümünde şu seçenekler vardır: ses dosyası yükleme, "Recognize lyrics" butonu, stil ve vokal
seçimi ve **Reference melody: Close ("Keep melody & structure") / Free ("Keep main melody only")**. Bu isimler projedeki
Yakın/Serbest ile neredeyse aynıdır.

### Koddan doğrulanan kısım

Depodaki cover akışı ([cover.py](https://github.com/NoizAI/YuE2-Turbo/blob/HEAD/src/yue2/cover.py),
[service.py](https://github.com/NoizAI/YuE2-Turbo/blob/HEAD/src/yue2/service.py)) önerilen yöntemle aynıdır:

1. `POST /v1/covers`: ses dosyası, stil, sözler, seed.
2. SheetSage2, `melody_only=True` ile şu tahminleri yapar: zaman damgaları, ölçü başları ve ölçü birimi, bölüm yapısı,
   ton, tam melodi (vokal + enstrüman). Akorlar atılır.
3. SheetSage2 GPU'dan atılır; YuE2 her zaman `cot="melody"` ile çalışır.

Kodda Demucs, Whisper ya da başka bir ASR yoktur. Sözler zorunludur ve kullanıcı tarafından yazılır. README, orijinal
şarkıcının sesinin kopyalanmadığını açıkça belirtir.

### Doğrulanamayan kısım

- **"Recognize lyrics":** Arkasında hangi modelin çalıştığı ve vokalin önce ayrılıp ayrılmadığı hiçbir yerde
  yazmaz. Model kartının önerdiği Qwen3-ASR veya Gemini kullanılıyor olabilir; doğrulanmadı.
- **Close / Free:** Sunucu kodu kapalı olduğu için iki olası yorum vardır:

| Site | Bu projede | Sitede olası karşılığı (tahmin) |
|---|---|---|
| Close | Akorlu nota + `cot=full` | Muhtemelen aynısı; resmi kılavuz sabit armoni için `cot=full` önerir |
| Free | Akorlar silinir; vokal ve enstrüman (`V: Ins`) melodileri kalır; `cot=melody` | (a) Açık kaynak koddaki akış, yani projeyle aynı. Ya da (b) SheetSage2'nin `melody_vocal` seçeneği: yalnızca vokal melodisi kalır, enstrüman melodileri de atılır. YuE2-Turbo'daki `transcribe.py` bu seçeneği destekler. |

(b) doğruysa sitenin Free modu projedeki Serbest'ten ayrılır. Serbest'te üretilen cover'larda eşliğin orijinale benzer
kaldığı gözlendi; olası nedeni notada kalan `V: Ins` satırlarıdır. Sitede Free ile üretilen bir cover'ın eşliği
orijinalden belirgin şekilde farklıysa bu (b)'yi destekleyen bir işaret olur.

---

## 9. Bilinen sınırlar ve notlar

- **İlerleme mesajı:** "Şarkı üret" otomatik analiz yaparken arayüz "Referans şarkı analiz ediliyor (vokal ayırma +
  melodi çıkarma)" yazar ([yue2_ui.py:766](../source/yue2_ui.py#L766)). SheetSage2 seçiliyken vokal ayırma yapılmaz;
  ifade Basit analizden kalmıştır.
- **Mimari şeması:** [ENTEGRASYON.md](ENTEGRASYON.md) içindeki mimari şeması SheetSage2 ve Whisper eklenmeden önce
  yazılmıştır; yalnızca YuE2 ve Demucs görünür.
- **Lisans:** SheetSage2 ve YuE2 ağırlıklarının lisansı **CC BY-NC 4.0**'dır (ticari olmayan kullanım). Demo site
  yayınlanırken göz önünde bulundurulmalıdır.
- **Transkripsiyon hataları cover'a geçer.** Nota ne kadar doğruysa cover orijinale o kadar benzer; resmi kılavuz da
  notanın kaynak dinlenerek kontrol edilmesini önerir.

---

## Kaynaklar

- [YuE2-3B model kartı](https://huggingface.co/M-A-P/YuE2-3B)
- [YuE2 cover kılavuzu (M-A-P)](https://github.com/multimodal-art-projection/YuE/blob/main/docs/covers.md)
- [YuE2 proje sitesi](https://map-yue2.github.io/)
- [SheetSage2 README (yerel kopya)](../../models/SheetSage2/README.md)
- [yue.noizai.net](https://yue.noizai.net/)
- [NoizAI/YuE2-Turbo](https://github.com/NoizAI/YuE2-Turbo)
