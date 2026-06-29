# Virtual Drag & Drop — Hand Tracking

Program Python untuk menyeret (drag) dan melepas (drop) objek virtual
menggunakan gerakan tangan lewat webcam, tanpa menyentuh layar.

---

## Instalasi

```bash
pip install opencv-python cvzone mediapipe numpy
```

## Cara Menjalankan

```bash
python main.py
```

## Kontrol

| Gerakan | Fungsi |
|---|---|
| Jari Telunjuk | Kursor / pointer |
| Telunjuk + Jari Tengah didekatkan | Grab / angkat kotak |
| Pisahkan kedua jari | Lepas kotak |
| Tekan `R` | Reset posisi semua kotak |
| Tekan `Q` atau `ESC` | Keluar program |

## Struktur Folder

```
VirtualDragDrop/
├── main.py      ← File utama
└── README.md    ← Panduan ini
```
