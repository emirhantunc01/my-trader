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

## Stratejiler

- `core-satellite`: Sermayenin buy-and-hold cekirdegi + taktik trend islemleri.
- `trend`: EMA, ADX, MACD, RSI, CMF ve hacim filtresiyle long-only trend sistemi.
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

## Not

Backtest sonucu gelecekteki performansi garanti etmez. Komisyon, slippage ve ertesi
bar acilisindan islem varsayimlari dahil edilmistir, ancak canli piyasada likidite,
spread, veri gecikmesi ve broker emir gerceklesmesi sonuclari degistirebilir.
