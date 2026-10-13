# YuE2 - UI

> Mimari, entegrasyon süreci ve donanım gereksinimleri: [docs/ENTEGRASYON.md](docs/ENTEGRASYON.md) ·
> Colab: [colab/COLAB_YOL_HARITASI.md](colab/COLAB_YOL_HARITASI.md)

YuE2 müzik üretim modeli için Gradio arayüzü ve model sunucusu.

**Modeller Colab'da (NVIDIA L4) çalıştırılır**, kurulum için [Colab yol haritası](colab/COLAB_YOL_HARITASI.md).
Mac'te arayüz geliştirmesi modelsiz test sunucusuyla yapılır:
`./start_model_server.sh --mock --outputs outputs/mock` ve `./start_ui_mac.sh`.

Arayüz, [YuE-UI](https://github.com/joeljuvel/YuE-UI)'ın (Apache-2.0) prompt akışından esinlenerek YuE2 için
sıfırdan yazıldı. YuE-UI'dan yalnızca etiket listesi (`top_200_tags.json`), örnek şarkı
(`examples/example.json`) ve proje kaydet/yükle formatı alındı. Lisans ve atıf: `LICENSE`, `NOTICE`.

| Dosya | Görevi | Ortam |
|---|---|---|
| `source/yue2_ui.py` | Gradio arayüzü (port 7860), modeli kendi içinde yüklemez | `yue-ui/venv` |
| `server/yue2_server.py` | `yue2.YuE2Pipeline`'ı HTTP API olarak sunar (port 7861) | `../yue_ortam` |
| `start_ui_mac.sh` | UI'ı başlatır | |
| `start_model_server.sh` | Model sunucusunu başlatır (`--mock` ile modelsiz test) | |
| `setup_windows.bat`, `start_model_server.bat`, `start_ui_windows.bat` | Windows'ta kurulum ve başlatma ([Windows](#windows)) | |
| `download_models.py` | Eksik model ağırlıklarını indirir (`setup_windows.bat` çağırır) | `../yue_ortam` |

## 1. UI'ı başlat
```bash
./start_ui_mac.sh
```
Tarayıcıda http://127.0.0.1:7860 adresini açın. Model sunucusu yokken üstte 🔴 durumu görünür. Bu normaldir.

## 2. Modeli başlat ve bağla
Ağırlıklar yerelde durur: `../models/YuE2-3B` (7.3 GB) ve `../models/YuE2-Vae` (0.5 GB). Bunlar bir kez
Hugging Face'ten indirildi. Sunucu bundan sonra tamamen çevrimdışı çalışır (`--offline`, `HF_HUB_OFFLINE=1`).

Ayrı bir terminalde:
```bash
./start_model_server.sh --preload
```
- UI 5 saniyede bir sunucuyu kontrol eder ve sunucu hazır olunca 🟢 durumuna geçer.
- Sık kullanılan seçenekler: `--device mps|cpu`, `--budget 16` (GiB), `--offload-ar`.
- Ağırlıklar başka bir klasördeyse: `YUE2_MODELS=/yol/models ./start_model_server.sh`

## Windows
NVIDIA ekran kartı gerekir. AMD veya Intel kartta model CPU'da çalışır, bu da kullanılamayacak kadar yavaştır.
Henüz gerçek bir Windows makinesinde denenmedi.

1. Kısa, boşluksuz bir klasör açıp (ör. `C:\yueModel`) depoyu içine `yue-ui` adıyla klonlayın:
   `git clone <depo adresi> yue-ui`. Ağırlıklar (`models`) ve ortamlar bu klasöre, `yue-ui`'nin yanına kurulur.
   Ağırlıklar depoda yoktur, kurulum indirir; Mac'teki `models` klasörünü `C:\yueModel\models` olarak
   kopyalarsanız yeniden inmez. Ortam klasörlerini (`yue_ortam`, `sheetsage_ortam`, `yue-ui/venv`, `.python`)
   **kopyalamayın**: Mac'e göre kuruludurlar, Windows'ta çalışmazlar.
2. Kurulum (bir kez, internet gerekir): `yue-ui` klasöründe `setup_windows.bat`. uv'yi, Python 3.12 + CUDA'lı
   torch 2.10 ortamını (`../yue_ortam`), eksik ağırlıkları ve SheetSage2 ortamını (`../sheetsage_ortam`) kurar.
   Tekrar çalıştırmak güvenlidir. SheetSage2 istenmezse `setup_windows.bat --no-sheetsage`.
3. İki ayrı pencerede: `start_model_server.bat --preload` ve `start_ui_windows.bat` (Gradio, 7860) ya da
   `frontend\start.bat` (web arayüzü, 7870). Modelsiz deneme: `frontend\start.bat --mock`.

**Model ağırlıkları:** `setup_windows.bat` 4. adımda `download_models.py`'yi çalıştırır ve eksik ağırlıkları
indirir (toplam ~18 GB). Hugging Face hesabı ya da token gerekmez. Bir model yalnızca dosyası eksikse iner;
yarıda kalan indirme, kurulum tekrar çalıştırılınca kaldığı yerden devam eder.

| Model | Klasör | Boyut | Ne için |
|---|---|---|---|
| YuE2-3B (`m-a-p/YuE2-3B`) | `models\YuE2-3B` | 7.3 GB | Şarkı üretimi |
| YuE2-VAE (`m-a-p/YuE2-Vae`) | `models\YuE2-Vae` | 0.5 GB | Sesi çözme |
| Whisper large-v3 turbo | `models\whisper-large-v3-turbo` | 1.6 GB | Dil algılama, Türkçe dışı sözler |
| Whisper large-v3 Turkish | `models\whisper-large-v3-turkish-general` | 3.1 GB | Türkçe sözler |
| Demucs (htdemucs) | `models\demucs` | 80 MB | Vokali ayırma (Cover) |
| SheetSage2 (`m-a-p/SheetSage2`) | `models\SheetSage2` | 0.2 GB | Nota çıkarma (Cover) |
| MERT-v2 (`m-a-p/MERT-v2-FullSong`) | `%USERPROFILE%\.cache\huggingface` | 5 GB | SheetSage2'nin temel modeli |

- Kurulumun geri kalanı bittiyse yalnızca indirme: `yue-ui` klasöründe
  `..\yue_ortam\Scripts\python.exe download_models.py ..\models --sheetsage`
- Mac'ten kopyalamak (internet yavaşsa): Mac'teki `yueModel/models` klasörünü `C:\yueModel\models` olarak
  kopyalayın. USB disk exFAT ya da NTFS olmalı; FAT32 4 GB'tan büyük dosya (YuE2-3B 7.3 GB) almaz. MERT-v2
  Hugging Face önbelleğinde durduğu için kurulumda yine iner.
- C: diski darsa MERT-v2'nin önbelleğini taşıyın: kurulumdan önce `setx HF_HOME D:\hf_cache`, sonra yeni bir
  pencere açın. Model sunucusu da aynı ayarla çalışmalı (`setx` kalıcıdır).

Windows'ta tek bir ortam (`yue_ortam`) hem model sunucusunu hem iki arayüzü çalıştırır (Colab'daki gibi).
Script'ler Python'u `PYTHONUTF8=1` ile başlatır, böylece Türkçe karakterler dosyalara ve loglara sorunsuz yazılır.

**Bellek:** `start_model_server.bat` sunucuyu `--budget auto` ile başlatır. Ayarlar ekran kartının belleğine göre
seçilir:

| Kart belleği | Ayar | Beklenen |
|---|---|---|
| 24 GB | `--budget 23` | Colab L4 gibi |
| 16 GB | `--budget 15 --offload-ar` | Sığar |
| 8-12 GB | `--offload-ar --phase-offload --shared-gpu-memory` | Sığar; uzun şarkı + CFG'de yavaşlayabilir |

Ağırlıklar (7.3 GB) ve şarkı aşamasının önbelleği (1.2-2.8 GB) 8 GB'lık karta birlikte sığmaz. YuE2-3B iki
yarıdan oluşur: token üretimi AR yarısını (katmanlar 2.8 GB, embedding ve LM head 1.5 GB), ses sentezi NAR
yarısını (2.8 GB) kullanır. `--phase-offload` her aşamada yalnızca gereken yarıyı karta koyar, diğer yarı RAM'de
bekler. En yüksek kullanım ~6 GiB'a iner. Hesaplama aynı kaldığı için kalite değişmez; aşama geçişleri iş başına
birkaç saniye sürer.

YuE2 PyTorch'u kartın belleğinin 2 GB altında sınırlar (8 GB kartta ~6 GiB); `--shared-gpu-memory` bu sınırı
kaldırır. Buna rağmen sığmayan kısmı NVIDIA'nın Windows sürücüsü paylaşımlı GPU belleğine (sistem RAM'i) koyar:
yavaşlar ama hata vermez. Bu, *CFG scale* 1'den büyükken ya da Nota planı *Kapalı* iken (önbellek iki katına
çıkar) uzun şarkılarda olabilir. Bunun için sürücü 536.40 ya da daha yeni olmalı ve NVIDIA Denetim Masası'ndaki
*CUDA - Sysmem Fallback Policy* ayarı *Driver Default* kalmalı. Monitör ekran kartına bağlıysa masaüstü ve
tarayıcı da kartın belleğini kullanır; sunucu açılışta kartta ne kadar yer boş olduğunu yazar.

## Gizlilik
- UI (7860) ve model sunucusu (7861) yalnızca `127.0.0.1`'i dinler, ağdaki başka cihazlar erişemez.
- Gradio'nun herkese açık link özelliği (`--share`) varsayılan olarak kapalıdır, analitik gönderimi kapatılmıştır.

Modelsiz test: `./start_model_server.sh --mock` kısa bir deneme tonu üretir ve tüm UI akışını doğrular.

## Text to Song: açıklama + söz
**✦ Text to Song** sekmesi Cover'dan bağımsızdır. MP3, nota çıkarma ve zaman çizelgesi yoktur. Melodiyi ve akorları
model kendisi besteler.

- **Şarkı açıklaması** (en çok 3000 karakter): tema, sahne, tempo, ruh hali.
- **Şarkı sözleri** (en çok 5000 karakter): bölümleri `[verse]`, `[chorus]`, `[bridge]` ile ayırın. `[Verse]` gibi
  büyük harfli etiketler küçük harfe çevrilir.
- **🎙 Vokal**: Varsayılan / Kadın / Erkek. Kadın ve Erkek seçimi etiketlere `female vocal` / `male vocal` ekler.
- **🎚 Tür**: hazır türler ve serbest *Özel tür* alanı.
- **Gelişmiş ayarlar**: CoT modu, seed, ⚡ Hızlı sentez.

Model tek bir `[Tags]` metni alır. Bu metin `türler, özel tür, vokal, açıklama` sırasıyla birleştirilir. Modele giden
metnin tamamı sonuç bölümündeki **Modele giden girdi** alanında görünür. Form girdileri işle birlikte saklanır.
**🗂 Geçmiş**'te ✦ ile işaretlenen bir üretim seçilip **Ayarları forma geri yükle**'ye basılırsa girdiler Text to Song
formuna geri gelir. *Rastgele seed* kapatılıp seed sabitlenirse aynı girdiyle tekrar üretip sonuçları karşılaştırabilirsiniz.

## Cover: MP3'ten yeni şarkı
Üstteki **🎵 Cover** sekmesini kullanın.
YuE2 sesi doğrudan girdi olarak alamaz, ama dışarıdan verilen bir ABC notayı takip edebilir. Cover akışı
referans şarkıyı bu notaya çevirir. Notayı kimin çıkaracağı **Nota çıkarıcı** seçeneğiyle belirlenir.

### SheetSage2 (varsayılan)
YuE2 ekibinin cover için kullandığı nota çıkarma modeli ([m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2),
CC BY-NC 4.0). Melodiyi, akorları (7'li, 6'lı dahil), vuruşları, tonu, bölümleri ve enstrüman melodisini (`V: Ins`)
doğrudan YuE2'nin formatında yazar.

- torch 2.8 istediği için kendi ortamında (`../sheetsage_ortam`, Python 3.11) ayrı süreç olarak çalışır
  (`server/sheetsage.py` → `server/sheetsage_runner.py`). Ses çözülmüş olarak verildiği için FFmpeg gerekmez.
- Sözler istenirse Demucs + Whisper yine çalışır; satırlar SheetSage2'nin bölümlerine yerleştirilir. Söz bulunmayan
  şarkılı bölüm `instrumental`, gerçek söz bulunan enstrümantal bölüm `verse` olarak yeniden adlandırılır, çünkü YuE2
  n'inci söz bloğunu n'inci şarkılı bölümde söyler.
- Vokalsiz kayıtlarda notada vokal melodisi olmaz; bunlar için Basit analiz'i kullanın.
- Mac'te CPU'da çalışır: 75 sn'lik parça ~30 sn. Kurulum (bir kez):
  ```bash
  python3 -m pip install uv
  uv venv --python 3.11 ../sheetsage_ortam
  uv pip install --python ../sheetsage_ortam/bin/python torch==2.8.0 torchaudio==2.8.0 \
      -r ../models/SheetSage2/requirements.txt
  ```
  Model ve dayandığı MERT-v2 (2.5 GB, SheetSage2'nin `config.json`'undaki sabit revizyon) bir kez indirilir.
  Sunucu çevrimdışı çalıştığı için MERT-v2 önceden Hugging Face önbelleğinde olmalı:
  ```bash
  hf download m-a-p/SheetSage2 --local-dir ../models/SheetSage2
  hf download m-a-p/MERT-v2-FullSong --revision d8ba1c745e733b3908ce6ad16ebeb17ac7600a42
  ```

### Basit analiz (`server/cover.py`)
1. Demucs (`htdemucs`, `../models/demucs`) ile vokal ve enstrümanlar ayrılır.
2. Tempo ve ton tahmin edilir, enstrümanlardan her ölçünün akoru çıkarılır.
3. Vokal melodisi (pYIN) 1/16'lık notalara oturtulur.
4. Sonuç, modelin kendi yazdığı ABC formatına çevrilir (`L:1/32`, `V: Vocal`/`V: Ins`, `% bölüm`).
   Bölüm adları sözlerdeki `[verse]`, `[chorus]` sırasından alınır.

Kullanım: Prompt sekmesinde referans şarkıyı yükleyin, etiketleri ve sözleri girin, **Şarkı üret**'e basın.
Melodi otomatik çıkarılır. Önce görmek isterseniz **Melodiyi çıkar (önizle)** butonunu kullanın. Çıkan nota
*Model ayarları → Harici ABC nota* alanına yazılır ve orada düzenlenebilir.

- Korunanlar: melodi, akorlar, tempo, bölüm yapısı. Korunmayanlar: orijinal ses, tını, aranjman.
- **Referans melodisi** (YuE2'nin cover rehberindeki iki mod): **Yakın** akorlu notayı CoT `full` ile gönderir.
  **Serbest** notadaki akor işaretlerini siler ve CoT `melody` kullanır; eşlik yeni stile göre kendi akorlarını
  seçer. Stil değiştirirken YuE2 Serbest'i öneriyor. Mod, analizden sonra da değiştirilebilir (nota elle
  düzenlenmediyse).
- Tempo tahmini bazen 4/3 veya 2 kat yanılabilir. Bu durumda **Tempo (BPM)** alanına doğru değeri girin.
- **Süre** varsayılanı 0, yani tüm şarkı. Hızlı bir deneme için **Başlangıç/Süre** ile bir bölüm seçin.
- Tahmini üretim süresi, sunucunun aynı GPU'daki önceki üretimlerinden hesaplanır (`GET /stats`); o cihazdaki
  ilk üretimden önce gösterilmez. Mac M4 için ölçülmüş değer (~6.5 sn işlem / ses saniyesi) yedek olarak kullanılır.
- Analiz çıktıları (`melody.wav`, `score.abc`, `lyrics.txt`) `outputs/uploads/<id>/analysis/` altında saklanır.
  Aynı dosya ve ayarlarla yapılan analiz önbellekten gelir.

### GPU paylaşımı
Üretim (YuE2) ile analiz (Demucs + Whisper) GPU'yu sırayla kullanır; ikisi birlikte 24 GB'a sığmaz.

- Üretim sürerken yeni bir analiz istenirse sunucu bekletmeden "bitince tekrar deneyin" der (409).
  Önbellekteki analizler GPU gerektirmediği için yine anında gelir.
- Analiz sürerken gönderilen üretim sıraya girer ve analiz biter bitmez başlar.
- Colab'da YuE2 üretimler arasında GPU'da tutulur (`--keep-loaded`, ~7.3 GB). Analizden önce GPU'da 8 GB'tan az
  boş yer varsa YuE2 bellekten çıkarılır ve sonraki üretimde yeniden yüklenir.
- Demucs ve Whisper her analizden sonra GPU'dan atılır.

### Sözleri tanı
YuE2'nin sözler için zaman girdisi yok: sözlerdeki n'inci `[bölüm]` bloğunu notadaki n'inci şarkılı `% bölüm`
üzerinde söyler. **✨ Sözleri tanı**, ayrılmış vokali Whisper'la kelime zamanlarıyla yazıya döker ve bölümleri
sözlerden çıkarır (tekrar eden blok `chorus`). Nota ve sözler aynı zaman çizelgesinden üretildiği için birebir
eşleşir; her satır orijinalde söylendiği ölçülere yerleşir.

- Sözlerin altındaki **zaman çizelgesi** her bölümün ve satırın şarkıdaki zamanını gösterir.
- Kelimeleri düzeltmek yapıyı bozmaz. Bölüm silinir/eklenirse arayüz eşleşmenin bozulduğunu uyarır.
- Dil otomatik algılanır (tüm şarkı için bir kez) ya da elle seçilir.

İki Whisper modeli kullanılır, bellekte aynı anda yalnızca biri bulunur:

| Model | Görevi | Boyut |
|---|---|---|
| `../models/whisper-large-v3-turbo` | Dili algılar; Türkçe dışındaki sözleri yazar | 1.5 GB |
| `../models/whisper-large-v3-turkish-general` | Türkçe algılanır ya da seçilirse sözleri yazar | 2.9 GB |

Türkçe model (turkmedstt, Apache-2.0), Cem Adrian - Uzun İnce Bir Yoldayım (canlı) kaydında kelime hatasını
%23.7'den %13.6'ya indirdi; Mac'te ~4 kat yavaş (bu şarkıda Whisper adımı ~4 dk) ve ~10 GB RAM kullanır.
Colab'da (CUDA) Whisper fp16 çalışır.
Türkçe model yoksa ya da `--whisper-tr ''` verilirse Türkçe şarkılar da turbo ile yazılır.

```bash
../yue_ortam/bin/hf download openai/whisper-large-v3-turbo --local-dir ../models/whisper-large-v3-turbo
../yue_ortam/bin/hf download turkmedstt/whisper-large-v3-turkish-general --local-dir ../models/whisper-large-v3-turkish-general
```

## Model ayarları
*Model ayarları* sekmesi YuE2'nin iş başına değiştirilebilen bütün ayarlarını içerir. Her iş, kullanılan ayarları
ilerleme alanında ve *Geçmiş*'te tek satırda gösterir (**Ayarlar:**); *Ayarları forma geri yükle* hepsini geri getirir.

| Ayar | Etkisi |
|---|---|
| CoT modu | `full` akorlu nota, `melody` akorsuz nota, `off` nota yok |
| Seed | Aynı seed ve aynı ayarlar aynı şarkıyı verir |
| CFG scale | 1'den büyük: stil etiketlerine ve sözlere daha sıkı uyum (notaya değil); şarkı aşaması ~2 kat yavaşlar |
| Ses sentezi adımı (ODE) | 4-64, varsayılan 32. ⚡ Hızlı sentez = 16 |
| FP8 (deneysel) | Yalnızca compute 8.9+ NVIDIA (ör. L4); kalite etkisi denenmedi, CUDA Graph kapandığı için yavaş olabilir |
| ABC ve Şarkı örneklemesi | Temperature, Top-p, Top-k, Tekrar cezası, ceza penceresi, Min/Max token |

Belleği ve arka ucu etkileyen ayarlar (`--budget`, `--offload-ar`, `--backend`) sunucu başlatılırken verilir;
müziği değiştirmezler. Süre tahminleri yalnızca 32 ve 16 adımlı üretimlerden hesaplanır.

## Çıktılar
Her üretim `outputs/<iş-id>/` klasörüne kaydedilir: `audio.mp3` (arayüzde çalan/indirilen), `audio.flac`, `score.abc`, `request.json`,
`config.json`, `result.json`, `job.json`. Geçmiş sekmesi bu klasörden beslenir ve sunucu yeniden
başlatıldığında da korunur.
