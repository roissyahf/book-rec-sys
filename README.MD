# Laporan Proyek Machine Learning II - Roissyah Fernanda Khoiroh

## Domain Proyek
Membaca dapat menambah pengetahuan baru, memperluas imaginasi, hingga melepas penat. Banyak cara untuk membaca buku, entah membaca buku secara fisik atau wujud digitalnya. Banyak bacaan juga yang bisa dipelajari, mulai dari fiksi maupun non-fiksi yang sama-sama memberikan manfaat.

UNESCO menyebutkan Indonesia memiliki minat baca yang rendah, hanya 0,001%, yang artinya dari 1000 orang Indonesia hanya 1 orang yang rajin membaca. Indonesia menduduki peringkat 60 dari 61 negara soal minat membaca, padahal Indonesia memiliki infrasturktur yang mendukung untuk kegiatan membaca (KOMDIGI, 2020).

Penyebab minat baca masih rendah dilasir dari [kompas.com](https://www.kompas.com/skola/read/2024/01/17/150723169/mengapa-minat-baca-di-indonesia-rendah-ini-penjelasannya?page=all#:~:text=Kurangnya%20minat%20baca%20di%20Indonesia,dan%20minimnya%20sarana%20perpustakaan%20sekolah.&text=Ini%20menjadi%20faktor%20utama%20penyebab%20minat%20baca%20siswa%20rendah.&text=Sementara%20itu%2C%20sekolah%20tidak%20selalu,kebiasaan%20membaca%20bagi%20para%20siswa) disebabkan oleh beberapa hal, di antaranya bahan bacaan yang kurang memikat, minimnya sarana perpustakaan sekolah sehingga minat untuk membaca menjadi kurang, sulit dalam menumbuhkan kebiasaan membaca bagi para siswa, belum menjadikan kegiatan membaca sebagai sumber utama memperoleh informasi, hingga masih tingginya angka buta huruf di Indonesia.

Di zaman yang serba digital ini, untuk meminjam atau membeli buku kita tidak perlu mengunjungi tempatnya karena semua bisa dilakukan secara online. Terkadang pengguna dihadapkan oleh berbagai macam pilihan buku sehingga kesulitan untuk memilih. Oleh karena itu, diperlukan sistem rekomendasi yang mampu memberikan rekomendasi judul buku kepada pengguna. Diharapkan, dengan diberikannya rekomendasi tersebut: minat pengguna untuk membaca meningkat, kebiasaan membaca dapat ditumbuhkan, sehingga dapat meningkatkan literasi masyarakat Indonesia.

Penelitian sebelumnya oleh Ilyasa & Yamasari (2023), membandingkan performa Cosine Similarity dan Euclidean Distance sebagai algoritma kesamaan item dalam merekomendasikan buku perpustakaan. Hasil penelitian menunjukkan bahwa Cosine Similarity mempunyai tingkat skor MAE yang cukup baik yaitu 0.647352 dibandingkan dengan Euclidean Distance yang memilki skor MAE yaitu 0.676872. Penelitian lainnya, mengombinasikan content based filtering (CBF), collaborative filtering (CF) dan association rule untuk menghasilkan rekomendasi buku di website penjualan buku online agar dihasilkan rekomendasi yang lebih efisien, efektif, dan sesuai dengan minat pembeli.

Projek ini akan bertujuan untuk membangun sistem rekomendasi yang dapat memprediksi judul buku yang cocok untuk suatu pengguna, menggunakan content-based filtering dan collaborative filtering untuk merekomendasikan judul buku yang disegani oleh pembaca lain dengan karakteristik yang mirip. Harapannya sistem ini mampu meningkatkan minat baca pengguna, serta mempermudah pengguna dalam memilih judul buku.

## Business Understanding

### Problem Statements
Bagaimana membangun sistem rekomendasi menggunakan content-based filtering dan collaborative filtering?

### Goals
Untuk memberikan pengalaman menyenangkan pada pengguna dengan diberikannya rekomendasi judul buku yang terpesonalisasi menggunakan content-based filtering dan collaborative filtering, sehingga pengguna tidak perlu bingung memikirkan judul buku mana yang akan dipilih. Ketika pengguna merasa puas maka platform perpustakaan online atau toko buku online juga akan diuntungkan karena keuntungan meningkat dan memperoleh feedback positif dari pengguna.

### Solution statements
- Pada content based filtering, akan digunakan algortima K-NN dengan pengukuran kemiripan Cosine Similarity untuk memberikan rekomendasi judul buku dengan konten serupa seperti memiliki genree, author yang sama.
- Pada collaborative filtering, akan digunakan Deep Learning dan SVD guna dihasilkan rekomendasi judul buku berdasarkan pengguna lain yang memiliki selera serupa yang diukur melalui rating, sehingga memungkinkan pengguna untuk memperoleh rekomendasi buku yang belum pernah mereka dengar namun selaras dengan minat mereka.

## Data Understanding
Proyek ini menggunakan dataset dari Kaggle yang berjudul *Book Recommendation Dataset* yang dapat diakses melalui tautan berikut https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data. Dataset tersebut berisikan 3 file csv, yaitu Books.csv, Ratings.csv, dan Users.csv dengan penjelasan detail terlampir pada tabel 1, tabel 2, dan tabel 3.

Tabel 1. Informasi detail dari Books.csv
Dataset ini mengandung informasi tentang buku, yang diidentifikasi berdasarkan ISBN.
| **Nama variabel** | **Uraian** | **Jumlah baris** | **Count** | **Dtype** |
| --- | --- | --- | -- | -- |
| ISBN | Kode pengidentifikasian buku yang bersifat unik, yang terdiri dari deretan angka 13 digit mengandung informasi: Judul buku, Penerbit, Kelompok penerbit | 271360 | non-null | object |
| Book-Title | Judul buku | 271360 | non-null | object |
| Book-Author | Penulis buku | 271358 | non-null | object |
| Year-Of-Publication | Tahun publikasi buku | 271360 | non-null | object |
| Publisher | Penerbit buku | 271358 | non-null | object |
| Image-URL-S | Link menuju gambar buku ukuran small | 271360 | non-null | object |
| Image-URL-M | Link menuju gambar buku ukuran medium | 271360 | non-null | object |
| Image-URL-L | Link menuju gambar buku ukuran large | 271357 | non-null | object |

Berbicara tentang kondisi data pada **Books.csv:**
- Kolom Image-URL-S, Image-URL-M, Image-URL-L didrop karena tidak diperlukan dalam pembuatan model.
- Terdapat 2 nilai kosong pada kolom Book-Author dan Book-Publisher, sehingga kedua baris tersebut didrop.
- Kolom Year-Of-Publication mengandung nilai yang tidak konsisten, terdapat 2 nama publisher pada kolom ini yang seharusnya tidak ada. Cara menanganinya yaitu menempatkan Publisher dan Year-Of-Publication sesuai dengan kolomnya, karena setelah diperiksa penempatan value tersebut terbalik.
- Terdapat tahun pada Year-Of-Publication yang ditulis dalam string, sehingga perlu diubah menjadi numerical value menggunakan fungsi `pd.to_numeric`. Untuk tahun yang tertulis 0 serta 2027 ke atas perlu diubah juga, caranya yaitu mengganti nilai tersebut dengan nilai kosong, lalu mengisi nilai kosong tadi dengan media pada kolom Year-Of-Publication.

Tabel 2. Informasi detail dari Ratings.csv
| **Nama variabel** | **Uraian** | **Jumlah baris** | **Count** | **Dtype** |
| --- | --- | --- | -- | -- |
| User-ID | ID pengguna yang telah dianonimkan dan dipetakan ke bilangan bulat | 1149780 | non-null | int64 |
| ISBN | Kode pengidentifikasian buku yang bersifat unik, yang terdiri dari deretan angka 13 digit mengandung informasi: Judul buku, Penerbit, Kelompok penerbit | 1149780 | non-null | object |
| Book-Rating | Rating buku yang diberikan oleh pengguna, baik secara eksplisit yang dinyatakan dalam skala 1-10 (nilai yang lebih tinggi menunjukkan apresiasi yang lebih tinggi), atau implisit dinyatakan dengan 0 | 1149780 | non-null | int64 |

Meskipun terdapat rating implisit yang ditandai dengan nilai 0, projek ini hanya menggunakan rating eksplisit dengan nilai 1-10 dengan alasan memberikan fokus pada model dengan hanya memproses rating eksplisit yang mengindikasikan kepuasan pengguna terhadap buku yang dibeli.

Tabel 3. Informasi detail dari Users.csv
| **Nama variabel** | **Uraian** | **Jumlah baris** | **Count** | **Dtype** |
| --- | --- | --- | -- | -- |
| User-ID | ID pengguna yang telah dianonimkan untuk menjaga privasi | 278858 | non-null | int64 |
| Location | Lokasi pengguna | 278858 | non-null | object |
| Age | Usia pengguna | 168096 | non-null | float64 |

Setelah ditelusuri, sebanyak 39.7% dari kolom Age isinya kosong. Selain itu, terdapat user dengan usia lebih dari 90 tahun dan user berusia kurang dari 5 tahun. Untuk mengatasi masalah tersebut, value akan diisi dengan nilai median dari kolom Age.

### Exploratory Data Analysis (EDA)
Gambar 1. Usia User
![0 age](https://github.com/user-attachments/assets/d3fd191e-0d5e-4467-9584-6c7564ed613f)

Gambar 2. Countplot Rating Buku
![1 countplot book rating](https://github.com/user-attachments/assets/f73b6301-71ce-482e-a7b3-5cc0071c00ae)

Gambar 3. Top 10 buku
![2 top 10 book title](https://github.com/user-attachments/assets/3ed87e2b-86ca-4c56-9104-a4cde3884d79)

Gambar 4. Top 10 author
![3  top 10 book author](https://github.com/user-attachments/assets/098140e8-3b96-4b3b-a88a-b2f0b3d7c342)

Gambar 5. Top 10 publisher
![4  top 10 book publisher](https://github.com/user-attachments/assets/11fe6b65-3c77-4c09-93f8-50688d6d2bb3)

Berdasarkan visualisasi pada Gambar 1, 2 dan 3, diperoleh informasi:
1. Persebaran usia pada dataset yang digunakan yaitu usia 30 mendominasi hingga mencapai angka 120000, kemudian disumbang oleh user dengan usia di bawah 30 tahun yaitu 15-29 tahun, dan user di atas usia 30 tahun dengan rentang usia 31-75 tahun. Hal ini mengindikasikan, user di platform tersebut sangat variatif karena merangkul remaja, orang dewasa, orang tua, hingga orang lanjut usia.
2. Pada projek ini hanya digunakan rating eksplisit. Rating 8 menjadi yang tertinggi mencapai angka 1800, lalu disusul oleh rating 10, 9, dan 7. Artinya, user puas dan suka dengan buku yang dibelinya. Adapun, masih ada user yang memberikan rating rendah yaitu rating 4, 3, 2 bahkan 1 meskipun jumlahnya jika ditotal kurang dari 250 orang. Perlu digali penyebabnya, sehingga ke depannya user yang memberi rating rendah dapat meningkatkan user experiencenya sehingga diharapkan dapat memberikan rating yang lebih tinggi.
3. Barchart tentang top 10 buku, author, dan publisher, memberikan temuan tentang buku, author, dan publisher yang popular. John Grisham menjadi author nomer 1, buku dengan judul 'The Lovely Bones: A Novel' menempati tangga teratas dalam jajaran buku populer, sedangkan publisher nomer 1 yaitu Ballantine Books.

## Data Preparation
### Tahapan persiapan untuk content-based filtering
Jumlah data pada Books.csv dan Ratings.cv yang mencapai ratusan ribu, akan membebani komputasi serta rawan terjadinya restart otomatis terhadap kernel. Oleh karena itu, diambil sampel secara acak dari kedua dataset tersebut, dengan hanya mengambil 0.1 dari keseluruhan dataset. Selanjutnya, dilakukan penggabungan data dari sample Books.csv dengan sample Ratings.csv, dengan menggunakan `merge` pada kolom ISBN. Diperoleh dataset hasil penggabungan dengan jumlah baris sebanyak 30055, dan jumlah kolom sebanyak 7.

Setelah itu, dataset hasil penggabungan tadi, ditransformasi ke dalam bentuk pivot table, dengan `index='Book-Title'`, `columns='User-ID'`, dan value yang mengisi sel-sel tersebut yaitu `Book-Rating`, dengan nilai nol akan mengisi sel-sel kosong. Sebelum memasuki model KNN, pivot table akan dikonversi ke bentuk sparse-matrix untuk menyimpan data yang berisi sejumlah besar elemen bernilai nol, sehingga dapat menghemat sejumlah besar memori dan mempercepat pemrosesan data. 

Dalam projek ini, juga diimplementasikan **content-based filtering dengan menghitung cosine-similarity**, tahap pemrosesannya melibatkan penggunaan TF-IDF (Term Frequency and Inverse Document Frequency). **TF-IDF** merupakan ukuran relevansi sebuah kata dengan dokumen, yang memiliki formula (Katsov, 2017):
```math
\[
TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)
\]
```

**Term-Frequency** merupakan frekuensi kemunculan term i pada dokumen j dibagi dengan total term pada dokumen j, dengan formula (Katsov, 2017):
```math
\[
TF(t, d) = \frac{f(t, d)}{\sum_{k}f(k, d)}
\]
```

Di mana:
- \(t\) adalah istilah (term) tertentu.
- \(d\) adalah dokumen tertentu.
- \(f(t, d)\) adalah frekuensi kemunculan istilah \(t\) dalam dokumen \(d\).
- \(\sum_{k}f(k, d)\) adalah jumlah semua istilah dalam dokumen \(d\).

sedangkan **Inverse Document Frequency** akan mengurangi bobot suatu term jika kemunculannya banyak tersebar diseluruh dokumen, memiliki formula (Katsov, 2017):
```math
\[
IDF(t) = \log\left(\frac{N}{1 + n(t)}\right)
\]
```

Di mana:
- \(t\) adalah istilah (term) tertentu.
- \(N\) adalah jumlah total dokumen dalam koleksi.
- \(n(t)\) adalah jumlah dokumen yang mengandung istilah \(t\).
- \(1\) ditambahkan ke \(n(t)\) untuk menghindari pembagian dengan nol.

Projek ini menghitung TF-IDF menggunakan `TfidfVectorizer()` untuk mengubah teks kolom `Book-Author` dari sample dataset menjadi representasi numerik berbasis frekuensi term dan pentingnya term dalam seluruh dataset. Matriks yang dihasilkan memiliki dimensi sesuai jumlah dokumen (baris) dan fitur (kolom), matriks ini kemudian dikonversi menjadi *DataFrame* Pandas di mana kolomnya mewakili fitur (kata) dan indeksnya adalah judul buku dari kolom `Book-Title`. Selanjutnya, kemiripan antar buku dihitung menggunakan cosine similarity.

### Tahapan persiapan untuk collaborative filtering dengan Deep Learning
Berikut tahapan yang diperlukan sebelum membangun model guna memprediksi *Book-Rating* berdasarkan kombinasi *User-ID* dan *ISBN*:
1. Seperti halnya pada content-based filtering, hanya digunakan sample data yang diambil secara acak dari Users.csv dan Ratings.csv untuk melatih model ini.
2. Melakukan encoding pada tipe data `object` yaitu `ISBN` dan `User-ID` sehingga model dapat mengolahnya.
3. Membuat fitur masukan yang dinamanakan `x` dengan menggabungkan kolom yang sudah di-*encode* (`User-ID-encoded` dan `ISBN-encoded`).
4. Untuk kolom target `y` (Book-Rating) dinormalisasi dalam rentang [0, 1] menggunakan nilai minimum dan maksimum rating.
5. Membagi dataset menjadi data training (80%) dan data validation (20%) berdasarkan indeks.

### Tahapan persiapan untuk collaborative filtering dengan SVD
Pada dasarnya, tahapannya sama seperti halnya saat membangun model sebelumnya yaitu dimulai dengan mengambil sampel data secara acak dari Ratings.csv dengan `frac=0.25`, serta melakukan encoding pada tipe data `object` yaitu `ISBN` dan `User-ID`. Selanjutnya, data diubah menjadi format yang dapat diproses oleh library `Surprise` menggunakan objek `Reader` dan fungsi `Dataset.load_from_df`. Dalam hal ini, dataset tidak dipisah ke dalam training set dan validation set, melainkan digunakan cross-validation dengan 10 fold.

## Model Development & Result
### Content-based filtering
Content-based filtering merekomendasikan item berdasarkan parameter atau ciri khas yang dimiliki item tersebut, seperti judul buku, author, genree. Keuntungan dari content-based filtering yakni, informasi untuk setiap perwakilan elemen data diketahui, sehingga memungkinkan pengguna memperoleh informasi yang relevan (Putra & Santika, 2020). Sedangkan content-based filtering juga memiliki kekurangan, yaitu hasil rekomendasinya hanya mengacu pada isi item rekomendasi, sehingga hasil rekomendasinya kurang beragam.

K-nearest neighbor (K-NN) adalah algoritma yang digunakan untuk menentukan tetangga (pengguna terdekat) mana yang paling relevan untuk menghasilkan rekomendasi yang dapat diandalkan, sehingga memungkinkan untuk memilih hanya k tetangga terbaik dengan nilai korelasi tertinggi (Hssina, Grota, & Erritali, 2021). Untuk merekomendasikan barang kepada pengguna V, K-NN akan mencari pengguna lain yang memiliki riwayat transaksi yang mirip dengan pengguna V. Setelah mendapatkan daftar pengguna yang mirip dengan pengguna V, K-NN mencari produk yang dibeli oleh pengguna yang mirip dengan pengguna V dan merekomendasikan produk yang belum pernah dibeli oleh pengguna V, diurutkan berdasarkan kriteria terlaris (Aditya, Budi, & Munajat, 2016).

Dalam memberikan rekomendasi, k-NN akan mencari item yang mirip yang kiranya disukai oleh pengguna. Similarity-score menjadi ukuran yang digunakan untuk menilai kemiripan item. Similarity-score berkisar antara 0 hingga 1, semakin tinggi nilainya semakin mirip kedua item tersebut. Anggap terdapat dua item, maka fungsi sim(i1, i2) dapat menggambarkan kemiripan kedua item tersebut (kim, 2019).

Proyek ini menggunakan cosine similarity untuk menghitung kemiripan antar buku yang akan direkomendasikan. Cosine similarity akan menghitung kesamaan antara dua vektor dengan menghitung sudut kosinusnya, serta melihat apakah kedua vektor menghadap ke arah yang sama. Sudut kosinusnya yang semakin kecil, mengindikasikan similarity-score yang besar antara kedua item, yang artinya kedua item semakin mirip. Berikut rumusnya:

```math
Cosine(A,B) = \frac{A \cdot B}{|A||B|}
```

Dimana:
- A.B adalah hasil perkalian dot antara vektor A dan vektor B
- |A| adalah norma (panjang) vektor A
- |B| adalah norma (panjang) vektor B

Cara mengimplementasikan K-NN cukup mudah, dengan memanggil `NearestNeighbors` dari library `sklearn.neighbors`. Untuk menemukan buku yang paling mirip yang ditandai dengan jaraknya yang dekat, model k-NN dibuat menggunakan metrik cosine-similarity dengan algoritma brute-force seperti pada code block berikut.

```
# create a NearestNeighbors model using cosine similarity and brute-force algorithm
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

# fit the model to the data (the sparse matrix)
model_knn.fit(df_matrix)
```

Gambar 6. Hasil rekomendasi buku dengan K-NN
![Screenshot 2024-11-09 143903 knn](https://github.com/user-attachments/assets/3ac22aaa-31d2-4d35-b3f6-08126ea66e1b)

### Collaborative filtering
Jenis collaborative filtering ini menggunakan ulasan pengguna pada item untuk menghasilkan rekomendasi. Digunakan teknik statistik untuk mengidentifikasi pengguna terdekat, pada set yang sama, elemen yang sama, peringkat yang mirip dengan pengguna aktif. Setelah pengguna terdekat diidentifikasi, collaborative filtering ini akan menggunakan algoritma yang berbeda untuk menggabungkan ulasan pengguna terdekat dan menghasilkan prediksi ke pengguna aktif (Lieberman, 1995). Ulasan yang biasanya digunakan yaitu rating. 

Keuntungan collaborative-filtering yaitu (1) tidak perlu _domain knowledge_ karena penyematannya dipelajari secara otomatis, (2) model dapat membantu pengguna menemukan minat baru karena pengguna yang sama tertarik pada item tersebut, (3) sistem tidak membutuhkan fitur kontekstual. Sedangkan kelemanahnnya yaitu (1) ketika diberikan model baru, sistem akan mengalami masalah _cold start problem_, atau kondisi dimana sistem tidak dapat membuat embedding untuk item tersebut dan tidak dapat meminta model dengan item ini karena item tersebut tidak terlihat selama training, (2) sulit untuk memasukkan fitur sampingan dari item yang direkomendasikan, contohnya untuk rekomendasi film, fitur sampingan dapat mencakup negara atau usia (Google for Developers, 2024).

Neural Collaborative Filtering (NCF) digunakan dalam proyek ini. Arsitektur NCF memiliki _input data sparse_ (rating pengguna dan peringkat item), yang diubah menjadi informasi _dense information_ menggunakan _embedding layer_. Multilayer perceptron kemudian menggabungkan kedua fitur. Dalam proses _training_, rating yang diketahui digunakan sebagai label. Setelah proses _training_ selesai, model dapat memprediksi rating yang tidak diketahui dalam proses nonlinier (Bobadilla dkk., 2020).

Pada projek ini, dibangun kelas `RecommenderNet` untuk membangun model NCF, dengan code block berikut.

```
EMBEDDING_SIZE = 50

class RecommenderNet(tf.keras.Model):

  # function initialization
  def __init__(self, num_users, num_books, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_books = num_books
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # user embedding layer
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-5)
    )
    self.user_bias = layers.Embedding(num_users, 1) # user_bias embedding layer
    self.books_embedding = layers.Embedding( # books_embeddings layer
        num_books,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-5)
    )
    self.book_bias = layers.Embedding(num_books, 1) # book_bias embedding layer

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # calling embedding layer 1
    user_bias = self.user_bias(inputs[:, 0]) # calling embedding layer 2
    book_vector = self.books_embedding(inputs[:, 1]) # calling embedding layer 3
    book_bias = self.book_bias(inputs[:, 1]) # calling embedding layer 4

    dot_user_book = tf.tensordot(user_vector, book_vector, 2)

    x = dot_user_book + user_bias + book_bias

    return tf.nn.sigmoid(x) # activation sigmoid

model = RecommenderNet(num_users, num_books, EMBEDDING_SIZE) # model init

# model compiling
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

Model ini menggunakan embedding untuk memetakan pengguna dan buku ke dalam ruang vektor berdimensi rendah. Terdapat dua embedding untuk masing-masing entitas: satu untuk pengguna (user) dan satu lagi untuk buku (books), keduanya memiliki vektor embedding berukuran 50 (ditentukan oleh `EMBEDDING_SIZE`). Setiap entitas juga memiliki bias terpisah. Dalam fungsi `call`, model menghitung interaksi antara vektor pengguna dan buku menggunakan operasi dot product, kemudian menjumlahkannya dengan bias pengguna dan buku. Hasilnya diproses melalui fungsi aktivasi sigmoid untuk memprediksi kemungkinan pengguna menyukai buku tersebut. Model ini dilatih dengan menggunakan binary cross-entropy sebagai fungsi loss dan dioptimalkan dengan Adam optimizer.

Gambar 7. Hasil rekomendasi buku dengan Deep Learning
![Screenshot 2024-11-09 140500 DL test](https://github.com/user-attachments/assets/086a79b7-8953-496a-9657-d1581107f6a8)

Projek ini juga menggunakan model lain yaitu Singular Value Decomposition (SVD) yang melakukan faktorisasi pada matriks bukan nol (Zheng, Ding, & Nie, 2018). Algoritma ini mengurangi jumlah fitur dari sebuah dataset dengan mengurangi dimensi ruang dari N-dimensi menjadi K-dimensi (di mana K < N), struktur matriks yang digunakan yaitu setiap baris mewakili pengguna, dan setiap kolom mewakili item. Elemen-elemen dari matriks ini adalah rating yang diberikan kepada item oleh pengguna.

Gambar 8. Hasil rekomendasi buku dengan SVD
![Screenshot 2024-11-09 143744 svd](https://github.com/user-attachments/assets/cff407aa-1c09-4042-9fe2-aac911d1c7a4)

## Evaluation
Kinerja K-NN dalam merekomendasikan buku, dapat dievaluasi melalui kemampuannya dalam memberikan rekomendasi yang sesuai bagi pengguna (Teknologi Informasi, Pascasarjana, & Teknologi Yogyakarta, 2022). Tahap evaluasi ini dilakukan dengan menghitung rata-rata skor akurasi untuk semua parameter k yang telah di-split  menggunakan K-Fold Cross Validation (Sarmandanaa, Widiarthaa, Putria, & Astawaa, 2024). Dengan formula skor akurasi yaitu (Utomo & Probosini, 2020):
```math
Accuracy = \frac{Total prediksi yang benar}{Total sampel}
```

Untuk model Deep Learning dan SVD, digunakan RMSE (*Root Mean Square Error*) sebagai metrik evaluasi, sebagai tambahan dihitung juga skor MAE (*Mean Absolute Error*) untuk mengevaluasi performa SVD. RMSE menghitung kesalahan prediksi dengan mengkuadratkan selisih nilai aktual dengan nilai prediksi, lalu dibagi dengan banyaknya data pada *set* yang digunakan, dan hasilnya diakarkan sebagaimana yang tercantum dalam formula berikut. Semakin kecil skor RMSE dan MAE, semakin baik performa model karena nilai prediksi yang dihasilkan semakin mendekati nilai aktual.
```math
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
```

```math
MAE = \sum_{i=1}^{D}|x_i-y_i|
```

Gambar 9. Hasil training Deep Learning model
![rmse dl](https://github.com/user-attachments/assets/ac4eb445-987c-4ce8-a961-ed314e420e13)

Menggunakan 30 epoch, hasil training dapat dilihat pada Gambar 6. Diperoleh RMSE pada training set sebesar 0.3583, sedangkan RMSE pada validation set sebesar 0.4284.

Tabel 2. Hasil training SVD
| Note | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean | Std  |   
|:-----|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|----:|
| RMSE (testset) |  3.1919 | 3.1725 | 3.1883 | 3.1817 | 3.1858 | 3.2060 |3.1733 | 3.1868 | 3.2092 | 3.1866 | 3.1882 | 0.0114  |
| MAE (testset)  |  2.7571 | 2.7535 | 2.7655 | 2.7625 | 2.7635 | 2.7875 |2.7418 | 2.7716 | 2.7905 | 2.7773 | 2.7671 | 0.0143  |
| Fit time       |   2.79  |  2.74  |  3.43  | 3.07   |  2.78  | 2.76   |3.90   |  2.84  |  2.85  | 2.89   |  3.00  |  0.36   |   
| Test time      |   0.06  |  0.34  |  0.11  |  0.07  |  0.06  |  0.06  |0.11   |  0.07  |  0.06  |  0.11  |  0.11  | 0.08    |

Dataset dipisah ke dalam 10 k-fold cross validation, diperoleh rata-rata skor RMSE sebesar 3.1882 sedangkan rata-rata skor MAE sebesar 2.7671. Hasil training menggunakan Deep Learning pada projek ini, lebih baik daripada dengan menggunakan SVD, mengingat skor RMSEnya lebih rendah dan memiliki selisih yang cukup jauh.

## Conclusion
Proyek ini telah menjawab problem statement, meraih goals, serta mencapai solution statement, mengingat sistem rekomendasi dengan content-based filtering dan collaborative filtering berhasil dibangun. Content-based filtering untuk merekomendasikan buku dapat dibangun dengan K-nearest neighbor (K-NN) dengan memanfaatkan Cosine Similarity untuk mengukur kemiripan antar item. Sedangkan, dalam membangun collaborative filtering dapat dilakukan dengan melatih model Deep Learning dan Singular Value Decomposition (SVD). Hasil rekomendasi telah terpersonalisasi, untuk K-NN pengguna diberikan rekomendasi buku yang mirip berdasarkan authornya. Sedangkan, untuk model Deep Learning dan SVD, hasil rekomendasi didasarkan pada pengguna lain yang memiliki selera serupa yang diukur melalui rating. Melalui hasil rekomendasi tersebut, pengguna menjadi terbantu dalam menemukan konten baru yang mungkin belum mereka lihat sebelumnya dengan mempersempit pilihan mereka.

Untuk mengukur kemampuan model dalam merekomendasikan item yang sesuai dengan minat pengguna, digunakan RMSE dan MAE. Dalam penelitian ini, tidak diukur ketepatan model K-NN dalam memberikan rekomendasi buku, namun jika membandingkan RMSE model Deep Learning dengan SVD, diperoleh hasil Deep Learning lebih unggul daripada SVD, dengan RMSE mencapai angka 0.428 pada validation set, sedangkan rerata RMSE di test set SVD masih di angka 3.1882.

## Referensi
Aditya, P. H., Budi, I., & Munajat, Q. (2016). A comparative analysis of memory-based and model-based collaborative filtering on the implementation of recommender system for e-commerce in Indonesia: a case Study PT X. Conference Paper. https://doi.org/10.1109/icacsis.2016.7872755.

Bobadilla, J., Alonso, S., & Hernando, A. (2020). Deep learning architecture for collaborative filtering recommender systems. Applied Sciences, 10(7), 2441. https://doi.org/10.3390/app10072441

Dewi, R. K. (2024). Mengapa Minat Baca di Indonesia Rendah? Ini Penjelasannya. https://www.kompas.com/skola/read/2024/01/17/150723169/mengapa-minat-baca-di-indonesia-rendah-ini-penjelasannya?.

Google for Developers. (2024). Collaborative filtering advantages & disadvantages. https://developers.google.com/machine-learning/recommendation/collaborative/summary.

Hssina, B., Grota, A., & Erritali, M. (2021). Recommendation system using the k-nearest neighbors and singular value decomposition algorithms. International Journal of Power Electronics and Drive Systems/International Journal of Electrical and Computer Engineering, 11(6), 5541. https://doi.org/10.11591/ijece.v11i6.pp5541-5548.

Ilyasa, M. Dzikri Hisyam., Yamasari, Yuni. (2023). Perbandingan Cosine Similarity Dan Euclidean Distance Pada Model Rekomendasi Buku Dengan Metode Item Based Collaborative Filtering. Journal of Informatics and Computer Science (JINACS). Vol. 04, No. 03.

Katsov, I. (2017). Introduction to Algorithmic Marketing: Artificial Intelligence for Marketing Operations.

Kim, Falk. (2019) "Practical Recommender Systems". Manning Publications. https://learning.oreilly.com/library/view/practical-recommender-systems/9781617292705/.

KOMDIGI. (2020). Kementerian Komunikasi dan Digital. https://www.komdigi.go.id/berita/pengumuman/detail/teknologi-masyarakat-indonesia-malas-baca-tapi-cerewet-di-medsos.

Lieberman, H. (1995). Letizia: an agent that assists web browsing. Proceedings of the 14th international joint conference on Artificial intelligence. Montreal, Canada. Vol. 1, pp. 924-9.

Mathew, P., Kuriakose, B. & Hegde., V. (2016). Book Recommendation System through content based and collaborative filtering method. 2016 International Conference on Data Mining and Advanced Computing (SAPIENCE). https.//doi.org/10.1109/SAPIENCE.2016.7684166.

Putra, A. I., Santika, R. R. (2020). Implementasi Machine Learning Dalam Penentuan Rekomendasi Musik Dengan Metode Content-Based Filtering. Edumatic: Jurnal Pendidikan Informatika. Vol. 4, No. 1. https://doi.org/10.29408/Edumatic.V4i1.2162.

Sarmandanaa, I M. T., Widiarthaa, I. M., Putria, L. A. A. R., Astawaa, I. G. S. (2024). Penerapan Metode Content Based Filtering Dan K-Nearest Neighbor Dalam Sistem Rekomendasi Musik. Jurnal Elektronik Ilmu Komputer Udayana. Vol. 13, No. 1. 

Teknologi Informasi, M., Pascasarjana, P., & Teknologi Yogyakarta, U. (2022). This Work Is Licensed Under A Creative Commons Attribution-Sharealike 4.0 International License Sistem Rekomendasi Pada Tokopedia Menggunakan Algoritma K-Nearest Neighbor. https://doi.org/10.31294/Jtk.V4i2.

Utomo, E.  P., & Probosini N. (2020). Pengaruh Sikap Dan Norma Subjektif Terhadap Intensitas Penggunaan Aplikasi Streaming Pada Generasi Z. Jurnal Ilmu Sosial Dan Humaniora. Vol. 9, No. 2, P. 241. https://doi.org/10.23887/Jish-Undiksha.V9i2.18581.

Zheng, S., Ding, C., Nie, F. (2018). Regularized Singular Value Decomposition and Application to Recommender System. No. 3. http://arxiv.org/abs/1804.05090.