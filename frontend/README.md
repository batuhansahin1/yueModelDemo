# YuE2 Web Arayüzü (frontend)

Gradio arayüzünün (`source/yue2_ui.py`) işlevlerini taşıyan, web sitesi olarak yayınlanabilecek demo arayüzü:
**✦ Text to Song**, **🎵 Cover**, iki modda da **Model ayarları**, **🗂 Geçmiş** ve **Sunucu**. Derleme adımı ve npm
paketi yok: düz HTML/CSS/JS ve yalnızca Python standart kütüphanesiyle yazılmış küçük bir sunucu.

```
Tarayıcı ──► frontend/server.py (:7870)      public/ (site) + /api vekili
                  │  HTTP, yalnızca sitenin kullandığı uç noktalar
                  ▼
             server/yue2_server.py (:7861)    YuE2, SheetSage2, Demucs, Whisper
```

Tarayıcı yalnızca web sunucusuyla konuşur. Model sunucusu `127.0.0.1`'de kapalı kalır, CORS ayarı gerekmez.

## Başlatma

Komutlar `yue-ui` klasöründen çalıştırılır; site http://127.0.0.1:7870 adresinde açılır.

| Ne | Komut |
|---|---|
| Modelsiz deneme (kısa bir deneme tonu üretir) | `./frontend/start.sh --mock` |
| Gerçek model | 1. terminal: `./start_model_server.sh --preload`, 2. terminal: `./frontend/start.sh` |
| Başka makinedeki model sunucusu | `./frontend/start.sh --api http://ADRES:7861` |

- Web sunucusu için Python 3.9+ yeterli, kurulum yok. `--mock`, Gradio arayüzünün ortamını (`yue-ui/venv`) kullanır.
- Web sunucusu açılırken model sunucusunun durumunu yazar ("bağlı" ya da "şu an kapalı"). Hangisi önce açılırsa
  açılsın sorun olmaz: model sunucusu sonradan başlarsa arayüz en geç 5 saniyede kendiliğinden bağlanır.
- Mock sunucusu ayrı bir portta (7862) çalışır, gerçek model sunucusuyla çakışmaz. Çıktıları `outputs/mock`
  altına gider ve web sunucusu kapanınca o da kapanır.
- Referans analizi (Demucs, Whisper, SheetSage2) gerçek modelde dakikalar sürebilir. Web sunucusu analizi kendisi
  bekler, sayfa da sonucu 1.5 saniyede bir sorar. Böylece Safari ya da önündeki bir vekil (nginx 60 sn, Cloudflare
  100 sn) uzun isteği kesmez. Önbellekteki analizler ve "GPU meşgul" (409) yanıtları yine anında gelir.
- Seçenekler: `--port`, `--host`, `--api`, `--password`, `--max-upload-mb`, `--analyze-wait`
  (`python3 frontend/server.py --help`). Ortam değişkenleri: `YUE2_SERVER`, `YUE2_WEB_PORT`, `YUE2_WEB_HOST`,
  `YUE2_WEB_PASSWORD`, `YUE2_MOCK_PORT`.
- ⌘/Ctrl + Enter açık olan modda şarkı üretir.

## Testler

`./frontend/test.sh` hepsini çalıştırır; model ya da GPU gerekmez. Tek adım: `./frontend/test.sh logic|server|e2e`.

| Adım | Ne doğrular | Gerekli |
|---|---|---|
| `logic` | `public/js/logic.js`: istekler, doğrulama, zaman çizelgesi, cover özeti, süre tahminleri | Node 18+ |
| `server` | Statik site, `/api` vekili, parola, yükleme sınırı, uzun analizin arka planda beklenmesi. Mock model sunucusuyla Text to Song, cover, iptal | `yue-ui/venv` |
| `e2e` | Headless Chrome'da tüm akış: üretim, model ayarları, cover (dosya, sözleri tanı, Yakın/Serbest; analiz arka plan yolundan), geçmiş, sayfa yenileme. Konsolda hata ya da CSP ihlali olmamalı | Chrome, Node 22+ |

Ekran görüntüleri için: `E2E_SCREENSHOTS=/tmp/yue2-ekran node frontend/tests/e2e.mjs`.

## Gradio arayüzüyle farklar

- **Model ayarları iki modda da var.** Gradio'da tam ayarlar yalnızca Cover'daydı; Text to Song'da CoT, seed ve hızlı
  sentez vardı. Her mod kendi ayarlarını tutar: Cover analizinin değiştirdiği CoT ve şarkı token sınırı Text to Song'a
  geçmez.
- **Referans melodisi varsayılanı Serbest** (Gradio'da Yakın).
- Cover sözlerindeki `[Verse]` gibi etiketler de küçük harfe çevrilir (Gradio bunu yalnızca Text to Song'da yapıyor).
- Sayfa yenilenince her modun son işi geri gelir; süren bir iş takip edilmeye devam eder.
- Model sunucusunun adresi tarayıcıdan değiştirilemez, web sunucusu başlatılırken `--api` ile verilir. Böylece
  yayındaki site başka adreslere istek atmak için kullanılamaz.
- Arayüz mantığı (`logic.js`) Gradio'nun fonksiyonlarından taşındı ve aynı girdilerle aynı metinleri üretir (zaman
  çizelgesi, cover özeti, süre tahminleri). Tek bilinçli fark: eski işlerdeki boş (`null`) örnekleme değerleri
  ayar satırında "değişti" olarak gösterilmez.

## Yayınlama

- `./frontend/start.sh --host 0.0.0.0 --password 'PAROLA'` siteyi ağa açar. Önüne TLS veren bir ters vekil koyun
  (Caddy, nginx, Cloudflare Tunnel): parola (HTTP Basic) HTTPS olmadan açık metin gider.
- Model sunucusu `127.0.0.1`'de kalmalı. Site ona yalnızca bilinen uç noktalarla ulaşır: `/health`, `/stats`,
  `/jobs…`, `/uploads`, `/analyze`.
- Siteyi açabilen herkes GPU kuyruğuna iş ekleyebilir ve **Geçmiş**'te bütün üretimleri (sözler dahil) görür.
  Yüklenen şarkılar `outputs/uploads/` altında kalır. Herkese açık bir demo için parola kullanın.
- Site dışarıdan hiçbir şey yüklemez (CDN, font, analitik yok); Content-Security-Policy başlığı bunu zorunlu kılar.
- Alt yolda da çalışır (`https://alan.com/yue2/`), çünkü bütün adresler görelidir.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `server.py` | Statik site, `/api` vekili, arka planda referans analizi ve `/data` (`../top_200_tags.json`, `../examples/example.json`) |
| `start.sh`, `test.sh` | Başlatma ve testler |
| `start.bat` | Windows'ta başlatma; `start.sh` ile aynı seçenekler (`--mock`, `--api`, `--host`, `--password`) |
| `public/index.html`, `public/styles.css` | Sayfa; açık/koyu tema, mobil düzen |
| `public/js/logic.js` | Gradio arayüzünden taşınan saf mantık (DOM yok, Node'da test edilir) |
| `public/js/api.js` | Model sunucusu API'si |
| `public/js/t2s.js`, `cover.js`, `history.js`, `main.js` | Görünümler ve yönlendirme |
| `public/js/settings.js`, `output.js`, `tags.js`, `ui.js` | Model ayarları, Üretim paneli, etiket alanı, yardımcılar |
| `tests/` | `logic.test.mjs`, `test_server.py`, `e2e.mjs` |
