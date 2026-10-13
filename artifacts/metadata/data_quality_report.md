# SAGE Dataset Audit and Quality Report

**Generated UTC:** 2026-10-13T09:44:19.440476+00:00  
**Dataset Source:** `tirtho149/SAGE`  
**Dataset Revision:** `main`  
**Selection Seed:** `42`  

## 1. Class Registry & Distribution

| Class ID | Class Name | Discovered | Train Count | Val Count | Test Count | Total Selected | Target Met |
|---|---|---|---|---|---|---|---|
| 0 | `Tomato::Tomato Leaf Curl Virus` | 50,027 | 799 | 172 | 172 | 1,143 | No |
| 1 | `Tomato::Bacterial Leaf Spot` | 27,205 | 800 | 171 | 172 | 1,143 | Yes |
| 2 | `Tomato::Late Blight` | 23,469 | 800 | 171 | 172 | 1,143 | Yes |
| 3 | `Rice::Blast` | 7,411 | 807 | 165 | 171 | 1,143 | Yes |
| 4 | `Rice::Brown Spot` | 5,495 | 797 | 169 | 177 | 1,143 | No |
| 5 | `Rice::Bacterial Leaf Blight` | 4,696 | 800 | 176 | 167 | 1,143 | Yes |
| 6 | `Corn::Common Rust` | 20,487 | 799 | 173 | 171 | 1,143 | No |
| 7 | `Corn::Northern Leaf Blight` | 10,764 | 800 | 171 | 172 | 1,143 | Yes |
| 8 | `Potato::Late Blight` | 11,615 | 800 | 171 | 172 | 1,143 | Yes |
| 9 | `Wheat::Rust` | 5,579 | 798 | 171 | 174 | 1,143 | No |
| 10 | `Mango::Anthracnose` | 2,600 | 800 | 171 | 172 | 1,143 | Yes |
| 11 | `Cauliflower::Black Rot` | 3,245 | 800 | 171 | 172 | 1,143 | Yes |
| 12 | `Papaya::Papaya Mosaic Virus` | 2,732 | 800 | 171 | 172 | 1,143 | Yes |
| 13 | `Sugarcane::Yellow Leaf` | 1,528 | 800 | 171 | 172 | 1,143 | Yes |
| 14 | `Banana::Panama Disease` | 1,465 | 800 | 171 | 172 | 1,143 | Yes |

## 2. Split Summary
- **Total Selected Records:** 17,145
- **Train Split (70%):** 12,000
- **Validation Split (15%):** 2,565
- **Test Split (15%):** 2,580

## 3. Data Integrity & Leakage Prevention
- **Group Leakage Check:** Group IDs are assigned per source + base filename.
- Group-level stratification guarantees that exact duplicates and near-duplicates do not cross splits.
- Zero image binary payloads are stored in this metadata manifest to ensure fast, scalable loading.
