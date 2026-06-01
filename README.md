# my-trader

Python tabanli algoritmik ticaret arastirma ve backtest araci.

Bu proje canli emir gondermez. Amaci stratejileri gercekci varsayimlarla test etmek,
Buy & Hold benchmark'i ile karsilastirmak ve risk metriklerini raporlamaktir.

## Kurulum

```powershell
python -m pip install -r requirements.txt
```

## Ornekler

Varsayilan core-satellite stratejisi:

```powershell
python main.py --symbols AAPL MSFT --period 5y --strategy core-satellite --walk-forward
```

Trend stratejisi:

```powershell
python main.py --symbols AAPL MSFT NVDA SPY --period 5y --strategy trend
```

Stratejileri ayni veri uzerinde karsilastirmak:

```powershell
python main.py --symbols AAPL MSFT --period 5y --compare
```

Mean-reversion stratejisi:

```powershell
python main.py --symbols AAPL MSFT --period 3y --strategy mean-reversion
```

Pairs trading:

```powershell
python main.py --strategy pairs --pair KO PEP --period 5y
```

CSV ve grafik raporu kaydetmek:

```powershell
python main.py --symbols AAPL --period 3y --strategy core-satellite --save-reports
```

MetaTrader 5 fiyat verisiyle sanal bakiye/paper trading:

```powershell
python main.py --paper --paper-source mt5 --symbols EURUSD --strategy trend --mt5-timeframe M15 --paper-balance 10000
```

MT5 terminali zaten acik ve hesaba giris yapmis durumdaysa genelde login bilgisi vermek gerekmez.
Belirli bir demo hesaba baglanmak icin:

```powershell
$env:MT5_PASSWORD="demo-password"
python main.py --paper --paper-source mt5 --symbols EURUSD GBPUSD --strategy trend --mt5-timeframe M15 --mt5-login 123456 --mt5-server "Broker-Demo"
Remove-Item Env:MT5_PASSWORD
```

Sanal portfoyu sifirlamak:

```powershell
python main.py --paper --paper-source mt5 --symbols EURUSD --paper-reset
```

MT5 olmadan paper motorunu hizli test etmek:

```powershell
python main.py --paper --paper-source yfinance --symbols AAPL MSFT --period 1y --strategy trend
```

## Stratejiler

- `core-satellite`: Sermayenin buy-and-hold cekirdegi + taktik trend islemleri.
- `trend`: EMA, ADX, MACD, RSI, CMF ve hacim filtresiyle long-only trend sistemi.
- `fast-trend`: MT5 demo denemeleri icin daha gevsek EMA/MACD trend sistemi.
- `scalp`: M1/M5 demo denemeleri icin daha hizli long/short sinyal sistemi.
- `mean-reversion`: Bollinger Bands, RSI, MFI, CMF ve z-score ile ortalamaya donus sistemi.
- `hybrid`: Trend ve mean-reversion sinyallerini birlestirir.
- `pairs`: Iki varlik arasindaki spread z-score'una dayali market-neutral arastirma modu.

## Raporlanan Metrikler

- Total return
- CAGR
- Volatility
- Max drawdown
- Sharpe
- Sortino
- Calmar
- Recovery factor
- Profit factor
- Win rate
- Market exposure
- Benchmark return
- Alpha vs benchmark
- Beta vs benchmark

## Durust Arastirma Kurallari

- Buy & Hold hala guclu bir benchmark'tir; botun onu her zaman gececegi varsayilmaz.
- Terminal raporu stratejinin benchmark'a gore getiri, drawdown ve Sharpe durumunu yorumlar.
- `--walk-forward` ile in-sample ve out-of-sample donemleri ayri gosterilir.
- `--compare` modu farkli stratejileri ayni semboller ve maliyet varsayimlariyla yan yana koyar.
- Pairs trading raporu, islem sonucundan once korelasyon ve z-score uygunluk kontrolu basar.
- Cok az islem, cok fazla islem veya zayif Sharpe gorulurse rapor bunu uyari olarak belirtir.

## MetaTrader 5 Paper Mode

- `--paper` modu MT5 terminalinden bar/tick verisi okuyabilir.
- Emirler gercek/demo hesaba gonderilmez; kod `order_send` kullanmaz.
- Sanal bakiye, acik pozisyonlar ve islem gecmisi `paper_state.json` dosyasinda tutulur.
- Strateji sinyali son kapanmis bar uzerinden hesaplanir; cari bid/ask fiyati sanal emir fiyati olarak kullanilir.
- Baslangic icin demo hesabi ve kucuk sembol listesi kullanmak daha guvenlidir.

## MetaTrader 5 Demo Emir Modu

Demo emir modu MT5 `order_send` kullanir, ancak sadece server adinda `Demo` gecen
hesaplarda calisir. Varsayilan hacim `0.01` lottur. Strateji sinyal uretmezse emir
gonderilmez.

```powershell
python main.py --execute-demo --confirm-demo-orders --symbols EURUSD --strategy trend --mt5-timeframe M15 --paper-bars 350 --demo-volume 0.01 --mt5-terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe"
```

Coklu sembol:

```powershell
python main.py --execute-demo --confirm-demo-orders --symbols EURUSD GBPUSD USDJPY --strategy trend --mt5-timeframe M15 --demo-volume 0.01 --mt5-terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe"
```

Surekli calistirmak:

```powershell
python main.py --execute-demo --confirm-demo-orders --symbols EURUSD GBPUSD USDJPY USDCAD SP500m --strategy trend --mt5-timeframe M15 --paper-bars 350 --demo-volume 0.01 --loop --sleep-seconds 1 --mt5-terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe"
```

Loop modu `Ctrl+C` ile durdurulur. Ayni sembolde botun actigi pozisyon zaten aciksa
yeniden alim emri yigilmaz; karar `POSITION_OPEN` olarak kalir. Demo emir modu
son gorulen sinyal durumunu `demo_state.json` dosyasinda tutar. Bu sayede ayni
kapanmis mumdaki buy sinyali tekrar tekrar islenmez; pozisyonu manuel kapatirsan
bot ayni eski sinyal yuzunden hemen yeniden almaz, yeni bir buy sinyali bekler.
Terminal ciktisinda `action` botun o dongude gercekten emir gonderip gondermedigini,
`signal` ise stratejinin son kapanmis mum icin BUY/SELL/NONE kararini gosterir.

Bot acildiginda zaten aktif olan buy sinyali varsayilan olarak izleme/warmup kabul
edilir. O sinyale de girmek istersen `--demo-trade-current-signal` ekleyebilirsin.
Hafizayi sifirlamak icin `--demo-reset` kullanilir. Kisa test icin:

```powershell
python main.py --execute-demo --confirm-demo-orders --symbols EURUSD --strategy trend --mt5-timeframe M15 --loop --sleep-seconds 1 --max-cycles 2 --mt5-terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe"
```

Emir gondermeden sinyal kosullarini gormek:

```powershell
python main.py --execute-demo --confirm-demo-orders --demo-dry-run --debug-signals --symbols EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD SP500m --strategy fast-trend --mt5-timeframe M15 --paper-bars 600 --max-cycles 1 --mt5-terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe"
```

Hizli demo scalper ornegi:

```powershell
python main.py --execute-demo --confirm-demo-orders --symbols AMD MSFT INTC NVDA --strategy scalp --mt5-timeframe M1 --paper-bars 600 --demo-volume 0.01 --demo-stop-atr 0.8 --demo-take-profit-atr 0.5 --demo-max-hold-minutes 3 --demo-allow-short --demo-trade-current-signal --demo-state demo_state_scalp_stocks.json --loop --sleep-seconds 3 --mt5-terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe"
```

Not: Bu mod gercek para hesabi icin tasarlanmamistir. Server adinda `Demo` yoksa
bilerek hata verir ve emir gondermez.

## Not

Backtest sonucu gelecekteki performansi garanti etmez. Komisyon, slippage ve ertesi
bar acilisindan islem varsayimlari dahil edilmistir, ancak canli piyasada likidite,
spread, veri gecikmesi ve broker emir gerceklesmesi sonuclari degistirebilir.
