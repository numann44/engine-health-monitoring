# CV project descriptions

Use only with the repository and release links that are actually available. Do not call this a deployed aircraft maintenance system.

## English

**Engine Health Monitoring — Predictive Maintenance & Robustness**  
Python · PyTorch · scikit-learn · Streamlit · NASA C-MAPSS

- Built a reproducible remaining-useful-life pipeline across four C-MAPSS scenarios (709 training and 707 test engines), using engine-disjoint validation and train-only operating-condition normalization.
- Trained 24 GRUs from scratch and 24 classical candidates; compared three-seed ensembles under missing, noisy, stuck and biased sensors with paired engine-bootstrap uncertainty.
- Implemented resumable training, checksum-verified inference, and an interactive sensor-stress dashboard. The predeclared robustness point criterion was met on 0/4 scenarios; retained negative results and failure analysis.

## Türkçe

**Engine Health Monitoring — Kestirimci Bakım ve Sensör Dayanıklılığı**

- NASA C-MAPSS'in dört senaryosunda 709 eğitim ve 707 test motoruyla, motor bazında ayrılmış doğrulama ve yalnız eğitimden öğrenilen normalizasyon kullanan kalan ömür tahmini hattı geliştirdim.
- Sıfırdan 24 GRU ve 24 klasik model adayı eğiterek eksik, gürültülü, takılı ve sapmış sensör koşullarını üç başlangıcın ortalaması ve motor bazında bootstrap belirsizliğiyle karşılaştırdım.
- Eğitime kaldığı yerden devam etme, checksum doğrulamalı model yükleme ve etkileşimli sensör deneyi paneli geliştirdim. Önceden belirlenen dayanıklılık nokta ölçütü 0/4 senaryoda sağlandı; olumsuz sonuçları ve hata analizlerini raporladım.
