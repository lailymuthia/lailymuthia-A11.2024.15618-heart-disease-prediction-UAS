const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, AlignmentType, ShadingType, PageBreak, LevelFormat,
  Header, Footer, PageNumber, TableOfContents, convertInchesToTwip
} = require("docx");
const fs = require("fs");

const PRIMARY = "6D28D9";
const DARK = "111827";
const GREY = "6B7280";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 200 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 150 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, size: 22, ...opts })],
    spacing: { after: 160 },
    alignment: AlignmentType.JUSTIFIED,
  });
}
function bullet(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 22 })],
    bullet: { level: 0 },
    spacing: { after: 80 },
  });
}
function img(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height } })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 150, after: 100 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 18, color: GREY })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 250 },
  });
}

function simpleTable(headers, rows) {
  const colWidth = 9000 / headers.length;
  const headerRow = new TableRow({
    children: headers.map(hd => new TableCell({
      width: { size: colWidth, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: PRIMARY },
      children: [new Paragraph({ children: [new TextRun({ text: hd, bold: true, color: "FFFFFF", size: 20 })] })],
    })),
  });
  const bodyRows = rows.map(r => new TableRow({
    children: r.map(cell => new TableCell({
      width: { size: colWidth, type: WidthType.DXA },
      children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20 })] })],
    })),
  }));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: headers.map(() => colWidth),
    rows: [headerRow, ...bodyRows],
  });
}

const OUT = "outputs";

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
  },
  sections: [
    // ===================== COVER =====================
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({ text: "", spacing: { before: 1800 } }),
        new Paragraph({
          children: [new TextRun({ text: "LAPORAN TEKNIS CAPSTONE PROJECT", bold: true, size: 40, color: PRIMARY })],
          alignment: AlignmentType.CENTER, spacing: { after: 200 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "PREDIKSI RISIKO PENYAKIT JANTUNG MENGGUNAKAN MACHINE LEARNING", bold: true, size: 30, color: DARK })],
          alignment: AlignmentType.CENTER, spacing: { after: 200 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "(Pendekatan CRISP-DM: Random Forest, XGBoost, dan SVM)", italics: true, size: 24, color: GREY })],
          alignment: AlignmentType.CENTER, spacing: { after: 1200 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Ujian Akhir Semester — Mata Kuliah Pembelajaran Mesin", size: 22 })],
          alignment: AlignmentType.CENTER, spacing: { after: 800 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Disusun oleh:", size: 22 })],
          alignment: AlignmentType.CENTER, spacing: { after: 80 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Laily Muthia N", bold: true, size: 26 })],
          alignment: AlignmentType.CENTER, spacing: { after: 40 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "NIM: A11.2024.15618", size: 22 })],
          alignment: AlignmentType.CENTER, spacing: { after: 40 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Program Studi Teknik Informatika", size: 22 })],
          alignment: AlignmentType.CENTER, spacing: { after: 1400 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "2026", size: 22, bold: true })],
          alignment: AlignmentType.CENTER,
        }),
      ],
    },
    // ===================== ISI LAPORAN =====================
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      headers: {
        default: new Header({ children: [new Paragraph({
          children: [new TextRun({ text: "Laporan Teknis — Prediksi Risiko Penyakit Jantung | Laily Muthia N (A11.2024.15618)", size: 16, color: GREY })],
          alignment: AlignmentType.CENTER,
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })],
          alignment: AlignmentType.CENTER,
        })] }),
      },
      children: [
        h1("1. PENDAHULUAN DAN LATAR BELAKANG"),
        h2("1.1 Latar Belakang"),
        p("Penyakit kardiovaskular merupakan salah satu penyebab kematian tertinggi di dunia. Deteksi dini terhadap risiko penyakit jantung dapat membantu tenaga medis melakukan intervensi lebih cepat dan menurunkan angka kematian akibat komplikasi yang tidak terdeteksi. Perkembangan teknik Machine Learning memungkinkan pembangunan sistem prediksi risiko berbasis data klinis pasien secara otomatis, cepat, dan dapat diskalakan."),
        p("Sebagai bagian dari Capstone Project Ujian Akhir Semester Mata Kuliah Pembelajaran Mesin, proyek ini mengimplementasikan pipeline Machine Learning end-to-end—mulai dari akuisisi data, eksplorasi, praproses, pemodelan, evaluasi, hingga deployment—menggunakan metodologi standar industri CRISP-DM (Cross-Industry Standard Process for Data Mining), guna mendemonstrasikan kemampuan menyelesaikan permasalahan dunia nyata secara komprehensif dan terukur."),
        p("Problem statement penelitian ini adalah: \u201cBagaimana membangun model klasifikasi yang mampu memprediksi apakah seorang pasien berisiko mengalami penyakit jantung (heart disease) berdasarkan data klinis dasar seperti usia, tekanan darah, kolesterol, hasil EKG, dan gejala terkait olahraga?\u201d Tujuan bisnisnya adalah menyediakan alat bantu skrining awal (decision support) yang murah dan cepat sebelum pasien menjalani pemeriksaan medis lanjutan yang lebih mahal (misalnya angiografi)."),

        h2("1.2 Tujuan"),
        bullet("Membangun pipeline Machine Learning lengkap mengikuti metodologi CRISP-DM."),
        bullet("Membandingkan performa tiga algoritma klasifikasi: Random Forest, XGBoost, dan SVM."),
        bullet("Melakukan tuning hyperparameter dan evaluasi menggunakan metrik yang relevan pada kasus medis (Accuracy, Precision, Recall, F1-Score, ROC-AUC)."),
        bullet("Menginterpretasikan model menggunakan Feature Importance dan SHAP agar hasil dapat dipahami oleh stakeholder non-teknis."),
        bullet("Mendeploy model terbaik dalam bentuk aplikasi web interaktif menggunakan Streamlit."),
        bullet("Metrik kesuksesan proyek: model dengan F1-Score di atas 0.85 pada data uji, serta aplikasi deployment yang berfungsi dan mudah digunakan."),

        h2("1.3 Sumber Data"),
        p("Dataset yang digunakan adalah Heart Failure Prediction Dataset, yang dipublikasikan oleh fedesoriano di platform Kaggle (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). Dataset ini merupakan gabungan dari lima dataset penyakit jantung klasik (Cleveland, Hungarian, Switzerland, Long Beach VA, dan Stalog Heart), berisi 918 baris data pasien unik dengan 11 fitur klinis dan 1 kolom target biner (HeartDisease: 0 = normal, 1 = berisiko)."),
        simpleTable(
          ["No", "Fitur", "Tipe Data", "Keterangan"],
          [
            ["1", "Age", "Numerik", "Usia pasien (tahun)"],
            ["2", "Sex", "Kategorikal", "Jenis kelamin (M/F)"],
            ["3", "ChestPainType", "Kategorikal", "Tipe nyeri dada (ATA, NAP, ASY, TA)"],
            ["4", "RestingBP", "Numerik", "Tekanan darah istirahat (mmHg)"],
            ["5", "Cholesterol", "Numerik", "Kadar kolesterol serum (mg/dl)"],
            ["6", "FastingBS", "Kategorikal", "Gula darah puasa > 120 mg/dl (1/0)"],
            ["7", "RestingECG", "Kategorikal", "Hasil EKG istirahat (Normal, ST, LVH)"],
            ["8", "MaxHR", "Numerik", "Detak jantung maksimal tercapai"],
            ["9", "ExerciseAngina", "Kategorikal", "Angina akibat olahraga (Y/N)"],
            ["10", "Oldpeak", "Numerik", "Depresi ST akibat olahraga"],
            ["11", "ST_Slope", "Kategorikal", "Kemiringan segmen ST puncak (Up, Flat, Down)"],
            ["12", "HeartDisease", "Target (Biner)", "0 = Normal, 1 = Heart Disease"],
          ]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),

        // ===== METODOLOGI =====
        h1("2. METODOLOGI (CRISP-DM)"),
        p("Penelitian ini mengikuti kerangka kerja CRISP-DM (Cross-Industry Standard Process for Data Mining) yang terdiri atas enam tahap iteratif: Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation, dan Deployment. Kerangka ini dipilih karena bersifat fleksibel, berorientasi pada tujuan bisnis, dan menjadi standar de-facto dalam proyek data science maupun machine learning."),

        h2("2.1 Business Understanding"),
        p("Tahap ini telah dijabarkan pada Bagian 1 — mendefinisikan problem statement, tujuan proyek, dan metrik kesuksesan berupa F1-Score minimal 0.85 pada data uji."),

        h2("2.2 Data Understanding (Exploratory Data Analysis)"),
        p("Eksplorasi data dilakukan untuk memahami struktur, kualitas, dan pola dalam data sebelum masuk tahap pemodelan. Lima insight utama yang ditemukan adalah sebagai berikut:"),
        bullet("Ketidakseimbangan kelas ringan: 55.3% pasien Heart Disease vs 44.7% Normal (rasio ±1.24:1), sehingga evaluasi tidak cukup hanya mengandalkan Accuracy."),
        bullet("Nilai 0 tidak wajar secara klinis ditemukan pada Cholesterol (18.7% data) dan RestingBP — mustahil secara medis dan diperlakukan sebagai hidden missing value."),
        bullet("Oldpeak (korelasi +0.40) dan MaxHR (korelasi -0.40) adalah fitur numerik paling berkorelasi dengan target."),
        bullet("ST_Slope bernilai 'Flat' dan ChestPainType bernilai 'ASY' (asymptomatic) sangat diskriminatif terhadap risiko penyakit jantung."),
        bullet("Tidak ditemukan multikolinearitas ekstrem (korelasi > 0.8) antar fitur numerik, sehingga seluruh fitur numerik aman digunakan bersama tanpa reduksi dimensi."),

        img(`${OUT}/01_distribusi_target.png`, 260, 195),
        caption("Gambar 1. Distribusi kelas target HeartDisease."),
        img(`${OUT}/05_heatmap_korelasi.png`, 400, 311),
        caption("Gambar 2. Heatmap korelasi antar fitur numerik."),
        img(`${OUT}/04_kategorikal_vs_target.png`, 470, 264),
        caption("Gambar 3. Distribusi fitur kategorikal terhadap target HeartDisease."),

        h2("2.3 Data Preparation"),
        p("Berdasarkan temuan EDA, dilakukan langkah-langkah praproses berikut:"),
        bullet("Data Cleaning: nilai Cholesterol dan RestingBP yang bernilai 0 diimputasi menggunakan median masing-masing kelas target, agar distribusi antar kelas tetap terjaga tanpa menghapus baris data."),
        bullet("Feature Engineering: menambahkan fitur turunan AgeGroup (kelompok usia) dan HR_Reserve (cadangan detak jantung, selisih estimasi HR maksimal teoritis 220-Age dengan MaxHR aktual)."),
        bullet("Encoding: One-Hot Encoding untuk lima fitur kategorikal (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)."),
        bullet("Scaling: StandardScaler untuk enam fitur numerik agar berada pada skala yang sebanding, penting khususnya untuk model SVM."),
        bullet("Data Splitting: dataset dibagi menjadi 70% data latih, 10% data validasi, dan 20% data uji, dengan stratifikasi pada target agar proporsi kelas tetap konsisten di setiap subset."),
        p("Seluruh proses di atas diimplementasikan sebagai Pipeline scikit-learn (ColumnTransformer + Estimator) agar konsisten diterapkan baik pada saat training maupun saat inference di aplikasi deployment, sehingga terhindar dari data leakage."),

        h2("2.4 Modelling"),
        p("Tiga algoritma klasifikasi dibangun dan dibandingkan:"),
        bullet("Random Forest — model ensemble berbasis bagging dari banyak decision tree, tahan terhadap overfitting dan mudah diinterpretasi."),
        bullet("XGBoost — model ensemble berbasis gradient boosting yang membangun tree secara sekuensial untuk memperbaiki kesalahan tree sebelumnya, umumnya memberikan akurasi tinggi pada data tabular."),
        bullet("Support Vector Machine (SVM) — model berbasis pencarian hyperplane pemisah optimal, efektif pada data berdimensi menengah dengan margin klasifikasi yang jelas."),
        p("Setiap model di-tuning menggunakan GridSearchCV dengan 5-fold cross-validation, dioptimalkan terhadap metrik F1-Score (karena adanya ketidakseimbangan kelas ringan). Grid hyperparameter yang diuji mencakup jumlah estimator dan kedalaman pohon (Random Forest, XGBoost), learning rate dan subsample (XGBoost), serta parameter C, kernel, dan gamma (SVM)."),

        h2("2.5 Evaluation"),
        p("Model dievaluasi pada data uji (test set) yang belum pernah dilihat selama training maupun tuning, menggunakan metrik Accuracy, Precision, Recall, F1-Score, dan ROC-AUC. Confusion Matrix dan ROC Curve digunakan sebagai analisis visual tambahan. Interpretasi model dilakukan menggunakan Feature Importance (bawaan model tree-based) dan SHAP (SHapley Additive exPlanations) untuk memahami kontribusi tiap fitur terhadap prediksi individual maupun global."),

        h2("2.6 Deployment"),
        p("Model terbaik disimpan dalam format pickle (.pkl) beserta pipeline preprocessing-nya, kemudian diintegrasikan ke dalam aplikasi web interaktif berbasis Streamlit yang terdiri dari lima halaman: Dashboard EDA, Model Demo (prediksi real-time), Evaluasi Model, Interpretasi Hasil, dan Dokumentasi."),

        // ===== HASIL DAN ANALISIS =====
        h1("3. HASIL DAN ANALISIS"),
        h2("3.1 Hasil Tuning Hyperparameter"),
        simpleTable(
          ["Model", "Best Params (ringkas)", "CV F1 (train)", "Val F1"],
          [
            ["Random Forest", "n_estimators=300, max_depth=5, min_samples_leaf=2", "0.878", "0.832"],
            ["XGBoost", "n_estimators=100, max_depth=3, lr=0.1, subsample=0.8", "0.887", "0.876"],
            ["SVM", "C=1, kernel=rbf, gamma=auto", "0.878", "0.843"],
          ]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),

        h2("3.2 Perbandingan Performa Model (Data Uji)"),
        img(`${OUT}/07_tabel_perbandingan_model.png`, 470, 110),
        caption("Gambar 4. Tabel perbandingan metrik evaluasi pada data uji."),
        p("Model XGBoost menunjukkan performa terbaik di seluruh metrik pada data uji, dengan F1-Score 0.910 dan ROC-AUC 0.943 — melampaui target metrik kesuksesan proyek (F1 ≥ 0.85). Hal ini konsisten dengan karakteristik XGBoost yang unggul dalam menangkap pola non-linear pada data tabular berukuran menengah seperti dataset ini."),

        img(`${OUT}/08_confusion_matrix.png`, 470, 141),
        caption("Gambar 5. Confusion Matrix ketiga model pada data uji."),
        img(`${OUT}/09_roc_curve.png`, 330, 283),
        caption("Gambar 6. ROC Curve perbandingan ketiga model."),

        h2("3.3 Interpretasi Model Terbaik (XGBoost)"),
        img(`${OUT}/10_feature_importance.png`, 400, 267),
        caption("Gambar 7. Feature Importance model XGBoost."),
        img(`${OUT}/11_shap_summary.png`, 330, 363),
        caption("Gambar 8. SHAP Summary Plot model XGBoost."),
        p("Analisis SHAP dan Feature Importance menunjukkan bahwa fitur ST_Slope, ChestPainType (khususnya kategori ASY/asymptomatic), Oldpeak, dan MaxHR adalah kontributor utama terhadap keputusan model. Nilai Oldpeak yang tinggi (indikasi ST depression akibat aktivitas fisik) secara konsisten mendorong prediksi ke arah 'berisiko', sedangkan nilai MaxHR yang tinggi mendorong prediksi ke arah 'normal'. Temuan ini sejalan dengan literatur medis mengenai indikator kardiovaskular, sehingga meningkatkan kepercayaan terhadap validitas model dari sudut pandang domain (bukan hanya statistik semata)."),

        h2("3.4 Aplikasi Deployment"),
        p("Model terbaik telah diintegrasikan ke dalam aplikasi Streamlit interaktif dengan lima halaman fungsional: (1) Dashboard EDA untuk eksplorasi data, (2) Model Demo untuk prediksi risiko pasien baru secara real-time melalui form input, (3) Evaluasi Model untuk membandingkan performa ketiga algoritma, (4) Interpretasi Hasil untuk menampilkan feature importance dan SHAP, serta (5) Dokumentasi proyek. Source code aplikasi tersedia pada file app/app.py dan dapat dijalankan secara lokal maupun di-deploy ke Streamlit Community Cloud."),

        // ===== KESIMPULAN =====
        h1("4. KESIMPULAN DAN REKOMENDASI"),
        h2("4.1 Kesimpulan"),
        bullet("Pipeline Machine Learning end-to-end berhasil dibangun mengikuti metodologi CRISP-DM secara lengkap, dari Business Understanding hingga Deployment."),
        bullet("Model XGBoost terpilih sebagai model terbaik dengan F1-Score 0.910 dan ROC-AUC 0.943 pada data uji, melampaui target metrik kesuksesan proyek."),
        bullet("Fitur klinis ST_Slope, ChestPainType, Oldpeak, dan MaxHR terbukti menjadi faktor paling berpengaruh terhadap prediksi risiko penyakit jantung, konsisten dengan pengetahuan medis yang ada."),
        bullet("Aplikasi Streamlit yang dibangun berfungsi sebagai alat bantu skrining awal yang interaktif, cepat, dan mudah digunakan oleh pengguna non-teknis."),

        h2("4.2 Keterbatasan Penelitian"),
        bullet("Dataset tidak memuat fitur SpO2 (kadar oksigen) dan tekanan darah diastolik, yang sebenarnya relevan secara klinis namun tidak tersedia pada sumber data ini."),
        bullet("Dataset merupakan gabungan dari beberapa rumah sakit/negara berbeda pada rentang waktu historis, sehingga karakteristik populasi mungkin tidak sepenuhnya merepresentasikan populasi pasien saat ini."),
        bullet("Model dilatih pada data tabular terstruktur; belum memasukkan data pencitraan medis (misalnya EKG mentah atau citra jantung) yang berpotensi meningkatkan akurasi prediksi."),

        h2("4.3 Rekomendasi"),
        bullet("Untuk pengembangan lebih lanjut, disarankan menambahkan data dari sumber lain yang memuat fitur SpO2 dan tekanan darah diastolik agar model lebih komprehensif."),
        bullet("Eksperimen dengan teknik ensemble stacking (menggabungkan Random Forest, XGBoost, dan SVM) berpotensi meningkatkan performa lebih lanjut."),
        bullet("Aplikasi deployment sebaiknya dilengkapi dengan mekanisme logging dan monitoring performa model secara berkala (model drift monitoring) apabila digunakan dalam skenario produksi nyata."),
        bullet("Hasil prediksi aplikasi ini bersifat sebagai alat bantu skrining awal (decision support), bukan pengganti diagnosis medis profesional, dan penggunaannya harus selalu didampingi evaluasi klinis oleh tenaga medis."),

        // ===== REFERENSI =====
        h1("5. REFERENSI"),
        p("fedesoriano. (2021). Heart Failure Prediction Dataset. Kaggle. https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction", { italics: false }),
        p("Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining."),
        p("Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32."),
        p("Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS)."),
        p("Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). CRISP-DM 1.0: Step-by-step Data Mining Guide. SPSS Inc."),
        p("Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("reports/Laporan_Teknis_UAS.docx", buf);
  console.log("Laporan berhasil dibuat.");
});
