# SignesiaResearch

Research project for optimizing spatio-temporal hand landmark representation from MediaPipe for preserving hand motion in 3D sign language avatars.

## Research Title

**Optimasi Representasi Spatio-Temporal Landmark MediaPipe untuk Preservasi Gerakan Tangan pada Avatar 3D Bahasa Isyarat**

## Overview

Penelitian ini merupakan bagian dari pengembangan sistem penerjemahan bahasa isyarat yang mengolah video bahasa isyarat menjadi representasi gerakan tangan dan selanjutnya diterjemahkan ke dalam avatar 3D.

Fokus penelitian ini adalah merancang representasi gerakan tangan secara **spatial dan temporal** sebelum digunakan pada proses motion mapping ke avatar 3D.

MediaPipe digunakan sebagai tahap ekstraksi landmark tangan dari video. Landmark yang diperoleh kemudian direpresentasikan berdasarkan hubungan spasial antar-landmark dan perubahan gerak antar-frame.

Berbeda dengan tahap skeleton preprocessing yang berfokus pada pembersihan dan perbaikan kualitas data skeleton, penelitian ini berfokus pada bagaimana informasi gerakan direpresentasikan dan dipertahankan ketika digunakan untuk animasi avatar 3D.

## Research Pipeline

```text
Video Bahasa Isyarat
        ↓
MediaPipe Hand Landmark Extraction
        ↓
Landmark Sequence
        ↓
Spatial Representation
        ↓
Temporal Representation
        ↓
Feature Optimization
        ↓
Motion Mapping
        ↓
Blender 3D Avatar
        ↓
Motion Preservation Evaluation
