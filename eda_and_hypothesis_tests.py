"""
DSA 210 Project — EDA & Hypothesis Tests
==========================================
Predicting Agricultural Crop Yield Across Turkish Provinces
Using Weather and Environmental Data

Author: Ceren Akkoyun
Date: April 2026

This script performs:
1. Data loading and overview
2. Exploratory Data Analysis (EDA) with visualizations
3. Statistical hypothesis tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings("ignore")

# Configure plot style for clean, professional visuals
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})
sns.set_theme(style="whitegrid", palette="colorblind")

FIGURES_DIR = "figures"

# ============================================================
# 1. DATA LOADING & OVERVIEW
# ============================================================
print("=" * 70)
print("1. VERİ YÜKLEME VE GENEL BAKIŞ")
print("=" * 70)

df = pd.read_csv("data/turkey_agriculture_dataset.csv")

print(f"\nVeri seti boyutu: {df.shape[0]} satır × {df.shape[1]} sütun")
print(f"\nSütunlar ve veri tipleri:")
print(df.dtypes.to_string())
print(f"\nİlk 5 satır:")
print(df.head().to_string())
print(f"\nTemel istatistikler:")
print(df.describe().round(2).to_string())
print(f"\nEksik değer sayısı:")
print(df.isnull().sum().to_string())

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 70)
print("2. KEŞİFSEL VERİ ANALİZİ (EDA)")
print("=" * 70)

# ----------------------------------------------------------
# 2.1 Target Variable Distribution (Class Balance)
# ----------------------------------------------------------
print("\n--- 2.1 Hedef Değişken Dağılımı (Sınıf Dengesi) ---")
class_counts = df["verim_sinifi_label"].value_counts()
print(f"Yüksek verim: {class_counts.get('Yüksek', 0)} gözlem")
print(f"Düşük verim:  {class_counts.get('Düşük', 0)} gözlem")
print(f"Oran: {class_counts.get('Yüksek', 0) / len(df) * 100:.1f}% / {class_counts.get('Düşük', 0) / len(df) * 100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ["#2ecc71", "#e74c3c"]
class_counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="black", alpha=0.85)
axes[0].set_title("Hedef Değişken Dağılımı\n(Verim Sınıfı: Yüksek vs Düşük)")
axes[0].set_xlabel("Verim Sınıfı")
axes[0].set_ylabel("Gözlem Sayısı")
axes[0].tick_params(axis="x", rotation=0)
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, colors=colors,
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 13})
axes[1].set_title("Sınıf Oranları")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/01_target_distribution.png")
plt.close()
print("  → Grafik kaydedildi: 01_target_distribution.png")

# ----------------------------------------------------------
# 2.2 Yield Distribution by Crop Type
# ----------------------------------------------------------
print("\n--- 2.2 Ürün Türüne Göre Verim Dağılımı ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
crops = df["urun"].unique()
for i, crop in enumerate(crops):
    ax = axes[i // 2][i % 2]
    crop_data = df[df["urun"] == crop]["verim_kg_dekar"]
    ax.hist(crop_data, bins=40, color=sns.color_palette("colorblind")[i],
            edgecolor="black", alpha=0.75)
    ax.axvline(crop_data.median(), color="red", linestyle="--", linewidth=2,
               label=f"Medyan: {crop_data.median():.0f}")
    ax.set_title(f"{crop} — Verim Dağılımı (kg/dekar)")
    ax.set_xlabel("Verim (kg/dekar)")
    ax.set_ylabel("Frekans")
    ax.legend()
    print(f"  {crop}: Ort={crop_data.mean():.1f}, Med={crop_data.median():.1f}, "
          f"Std={crop_data.std():.1f}, Min={crop_data.min():.1f}, Max={crop_data.max():.1f}")

plt.suptitle("Ürün Türlerine Göre Verim Dağılımları", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/02_yield_distribution_by_crop.png")
plt.close()
print("  → Grafik kaydedildi: 02_yield_distribution_by_crop.png")

# ----------------------------------------------------------
# 2.3 Yield Distribution by Region (Boxplot)
# ----------------------------------------------------------
print("\n--- 2.3 Bölgelere Göre Verim Karşılaştırması ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
region_order = ["Marmara", "Ege", "Akdeniz", "İç Anadolu", "Karadeniz",
                "Doğu Anadolu", "Güneydoğu Anadolu"]

for i, crop in enumerate(crops):
    ax = axes[i // 2][i % 2]
    crop_df = df[df["urun"] == crop]
    sns.boxplot(data=crop_df, x="bolge", y="verim_kg_dekar",
                order=region_order, ax=ax, palette="Set2")
    ax.set_title(f"{crop} — Bölgelere Göre Verim")
    ax.set_xlabel("")
    ax.set_ylabel("Verim (kg/dekar)")
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("Coğrafi Bölgelere Göre Ürün Verimleri", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/03_yield_by_region_boxplot.png")
plt.close()
print("  → Grafik kaydedildi: 03_yield_by_region_boxplot.png")

for crop in crops:
    crop_df = df[df["urun"] == crop]
    region_means = crop_df.groupby("bolge")["verim_kg_dekar"].mean().sort_values(ascending=False)
    print(f"\n  {crop} — Bölge ortalamaları (kg/dekar):")
    for region, val in region_means.items():
        print(f"    {region}: {val:.1f}")

# ----------------------------------------------------------
# 2.4 Weather Variables Distributions
# ----------------------------------------------------------
print("\n--- 2.4 Hava Durumu Değişkenleri Dağılımı ---")

weather_cols = ["ort_sicaklik_C", "toplam_yagis_mm", "bagil_nem_pct", "kuraklik_indeksi"]
weather_labels = ["Ort. Sıcaklık (°C)", "Toplam Yağış (mm)", "Bağıl Nem (%)", "Kuraklık İndeksi"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, (col, label) in enumerate(zip(weather_cols, weather_labels)):
    ax = axes[i // 2][i % 2]
    # Use province-level means (avoid crop duplication)
    prov_data = df.groupby(["il", "yil"])[col].first().reset_index()
    ax.hist(prov_data[col], bins=35, color=sns.color_palette("coolwarm", 4)[i],
            edgecolor="black", alpha=0.75)
    ax.set_title(f"{label} Dağılımı")
    ax.set_xlabel(label)
    ax.set_ylabel("Frekans")

plt.suptitle("Meteorolojik Değişkenlerin Dağılımları\n(İl-Yıl Bazında)",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/04_weather_distributions.png")
plt.close()
print("  → Grafik kaydedildi: 04_weather_distributions.png")

# ----------------------------------------------------------
# 2.5 Correlation Heatmap
# ----------------------------------------------------------
print("\n--- 2.5 Korelasyon Matrisi ---")

numeric_cols = ["verim_kg_dekar", "ort_sicaklik_C", "toplam_yagis_mm",
                "yagisli_gun", "bagil_nem_pct", "kuraklik_indeksi",
                "yagis_sapma_zscore", "sicaklik_genlik", "hasat_orani",
                "verim_degisim_pct"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            vmin=-1, vmax=1)
ax.set_title("Sayısal Değişkenler Arası Korelasyon Matrisi", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/05_correlation_heatmap.png")
plt.close()

print("  En güçlü korelasyonlar (verim ile):")
verim_corr = corr["verim_kg_dekar"].drop("verim_kg_dekar").abs().sort_values(ascending=False)
for col, val in verim_corr.head(5).items():
    direction = "+" if corr.loc["verim_kg_dekar", col] > 0 else "-"
    print(f"    {col}: {direction}{val:.3f}")
print("  → Grafik kaydedildi: 05_correlation_heatmap.png")

# ----------------------------------------------------------
# 2.6 Weather vs Yield Scatter Plots
# ----------------------------------------------------------
print("\n--- 2.6 Hava Durumu — Verim İlişkisi (Scatter) ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
wheat_df = df[df["urun"] == "Buğday"]

# Temperature vs Yield
axes[0].scatter(wheat_df["ort_sicaklik_C"], wheat_df["verim_kg_dekar"],
                alpha=0.3, c=wheat_df["verim_sinifi"], cmap="RdYlGn", edgecolors="gray", s=30)
axes[0].set_xlabel("Ortalama Sıcaklık (°C)")
axes[0].set_ylabel("Buğday Verimi (kg/dekar)")
axes[0].set_title("Sıcaklık vs Buğday Verimi")

# Rainfall vs Yield
axes[1].scatter(wheat_df["toplam_yagis_mm"], wheat_df["verim_kg_dekar"],
                alpha=0.3, c=wheat_df["verim_sinifi"], cmap="RdYlGn", edgecolors="gray", s=30)
axes[1].set_xlabel("Toplam Yağış (mm)")
axes[1].set_ylabel("Buğday Verimi (kg/dekar)")
axes[1].set_title("Yağış vs Buğday Verimi")

# Drought Index vs Yield
axes[2].scatter(wheat_df["kuraklik_indeksi"], wheat_df["verim_kg_dekar"],
                alpha=0.3, c=wheat_df["verim_sinifi"], cmap="RdYlGn", edgecolors="gray", s=30)
axes[2].set_xlabel("Kuraklık İndeksi")
axes[2].set_ylabel("Buğday Verimi (kg/dekar)")
axes[2].set_title("Kuraklık İndeksi vs Buğday Verimi")

plt.suptitle("Buğday: Meteorolojik Değişkenler ile Verim İlişkisi\n(Yeşil=Yüksek, Kırmızı=Düşük)",
             fontsize=14, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/06_weather_vs_yield_scatter.png")
plt.close()
print("  → Grafik kaydedildi: 06_weather_vs_yield_scatter.png")

# ----------------------------------------------------------
# 2.7 Yearly Yield Trends
# ----------------------------------------------------------
print("\n--- 2.7 Yıllara Göre Verim Trendleri ---")

fig, ax = plt.subplots(figsize=(14, 6))
for crop in crops:
    yearly = df[df["urun"] == crop].groupby("yil")["verim_kg_dekar"].mean()
    ax.plot(yearly.index, yearly.values, marker="o", linewidth=2, label=crop)

ax.set_title("Türkiye Geneli — Yıllara Göre Ortalama Verim Trendi", fontsize=15, fontweight="bold")
ax.set_xlabel("Yıl")
ax.set_ylabel("Ortalama Verim (kg/dekar)")
ax.legend(title="Ürün")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/07_yearly_yield_trends.png")
plt.close()
print("  → Grafik kaydedildi: 07_yearly_yield_trends.png")

# ----------------------------------------------------------
# 2.8 Regional Yield Heatmap (Region × Year)
# ----------------------------------------------------------
print("\n--- 2.8 Bölge × Yıl Verim Isı Haritası ---")

wheat_pivot = df[df["urun"] == "Buğday"].pivot_table(
    values="verim_kg_dekar", index="bolge", columns="yil", aggfunc="mean"
)
wheat_pivot = wheat_pivot.reindex(region_order)

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(wheat_pivot, annot=True, fmt=".0f", cmap="YlGn",
            linewidths=0.5, ax=ax)
ax.set_title("Buğday — Bölge × Yıl Ortalama Verim (kg/dekar)", fontsize=15, fontweight="bold")
ax.set_xlabel("Yıl")
ax.set_ylabel("Bölge")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/08_region_year_heatmap.png")
plt.close()
print("  → Grafik kaydedildi: 08_region_year_heatmap.png")


# ============================================================
# 3. HİPOTEZ TESTLERİ
# ============================================================
print("\n" + "=" * 70)
print("3. HİPOTEZ TESTLERİ")
print("=" * 70)

# ----------------------------------------------------------
# TEST 1: Independent Samples t-test
# Question: Do provinces with above-average rainfall produce
# significantly higher wheat yields?
# ----------------------------------------------------------
print("\n" + "-" * 60)
print("TEST 1: Bağımsız Örneklem t-Testi")
print("-" * 60)
print("Soru: Ortalamanın üzerinde yağış alan illerde buğday verimi,")
print("      ortalamanın altında yağış alan illere kıyasla anlamlı")
print("      şekilde daha yüksek midir?")
print()

wheat = df[df["urun"] == "Buğday"].copy()
rain_median = wheat["toplam_yagis_mm"].median()
high_rain = wheat[wheat["toplam_yagis_mm"] >= rain_median]["verim_kg_dekar"]
low_rain = wheat[wheat["toplam_yagis_mm"] < rain_median]["verim_kg_dekar"]

t_stat, p_value = stats.ttest_ind(high_rain, low_rain, equal_var=False)

print(f"  H₀: μ_yüksek_yağış = μ_düşük_yağış (İki grubun verim ortalaması eşittir)")
print(f"  H₁: μ_yüksek_yağış ≠ μ_düşük_yağış (İki grubun verim ortalaması farklıdır)")
print(f"")
print(f"  Yüksek yağış grubu:  n={len(high_rain)}, ort={high_rain.mean():.2f}, std={high_rain.std():.2f}")
print(f"  Düşük yağış grubu:  n={len(low_rain)}, ort={low_rain.mean():.2f}, std={low_rain.std():.2f}")
print(f"")
print(f"  Welch t-testi istatistiği: t = {t_stat:.4f}")
print(f"  p-değeri: p = {p_value:.6f}")
print(f"  Anlamlılık düzeyi: α = 0.05")
print(f"")
if p_value < 0.05:
    print(f"  ✓ SONUÇ: p < 0.05 → H₀ REDDEDİLDİ.")
    print(f"    Yüksek yağışlı illerde buğday verimi istatistiksel olarak")
    print(f"    anlamlı şekilde {'daha yüksek' if t_stat > 0 else 'daha düşük'}tir.")
else:
    print(f"  ✗ SONUÇ: p ≥ 0.05 → H₀ reddedilemez.")
    print(f"    İki grup arasında anlamlı bir fark bulunamadı.")

# Visualization for Test 1
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(high_rain, bins=30, alpha=0.65, label=f"Yüksek Yağış (n={len(high_rain)})", color="#3498db", edgecolor="black")
axes[0].hist(low_rain, bins=30, alpha=0.65, label=f"Düşük Yağış (n={len(low_rain)})", color="#e74c3c", edgecolor="black")
axes[0].axvline(high_rain.mean(), color="#3498db", linestyle="--", linewidth=2)
axes[0].axvline(low_rain.mean(), color="#e74c3c", linestyle="--", linewidth=2)
axes[0].set_title("Buğday Verim Dağılımı — Yağış Gruplarına Göre")
axes[0].set_xlabel("Verim (kg/dekar)")
axes[0].set_ylabel("Frekans")
axes[0].legend()

sns.boxplot(data=wheat, x=wheat["toplam_yagis_mm"] >= rain_median,
            y="verim_kg_dekar", ax=axes[1], palette=["#e74c3c", "#3498db"])
axes[1].set_xticklabels(["Düşük Yağış", "Yüksek Yağış"])
axes[1].set_title(f"t-Testi Sonucu: t={t_stat:.2f}, p={p_value:.4f}")
axes[1].set_xlabel("Yağış Grubu")
axes[1].set_ylabel("Verim (kg/dekar)")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/09_ttest_rainfall_yield.png")
plt.close()
print("  → Grafik kaydedildi: 09_ttest_rainfall_yield.png")

# ----------------------------------------------------------
# TEST 2: One-Way ANOVA
# Question: Is there a statistically significant difference in
# wheat yield across Turkey's 7 geographical regions?
# ----------------------------------------------------------
print("\n" + "-" * 60)
print("TEST 2: Tek Yönlü ANOVA (One-Way ANOVA)")
print("-" * 60)
print("Soru: Türkiye'nin 7 coğrafi bölgesi arasında buğday veriminde")
print("      istatistiksel olarak anlamlı bir fark var mıdır?")
print()

region_groups = [group["verim_kg_dekar"].values
                 for _, group in wheat.groupby("bolge")]
region_names = list(wheat.groupby("bolge").groups.keys())

f_stat, p_value_anova = stats.f_oneway(*region_groups)

print(f"  H₀: μ₁ = μ₂ = μ₃ = μ₄ = μ₅ = μ₆ = μ₇ (Tüm bölge ortalamaları eşittir)")
print(f"  H₁: En az bir bölgenin ortalaması farklıdır")
print(f"")
print(f"  Bölge ortalamaları:")
region_means = wheat.groupby("bolge")["verim_kg_dekar"].agg(["mean", "std", "count"])
for region in region_order:
    if region in region_means.index:
        r = region_means.loc[region]
        print(f"    {region:25s}: ort={r['mean']:.1f}, std={r['std']:.1f}, n={int(r['count'])}")
print(f"")
print(f"  F-istatistiği: F = {f_stat:.4f}")
print(f"  p-değeri: p = {p_value_anova:.2e}")
print(f"  Anlamlılık düzeyi: α = 0.05")
print(f"")
if p_value_anova < 0.05:
    print(f"  ✓ SONUÇ: p < 0.05 → H₀ REDDEDİLDİ.")
    print(f"    Bölgeler arasında buğday veriminde istatistiksel olarak")
    print(f"    anlamlı bir fark bulunmaktadır.")
else:
    print(f"  ✗ SONUÇ: p ≥ 0.05 → H₀ reddedilemez.")

# Visualization for Test 2
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=wheat, x="bolge", y="verim_kg_dekar",
            order=region_order, palette="Set2", ax=ax)
ax.set_title(f"ANOVA Sonucu: F={f_stat:.2f}, p={p_value_anova:.2e}\n"
             f"Buğday Verimi — Bölgelere Göre Karşılaştırma",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Coğrafi Bölge")
ax.set_ylabel("Verim (kg/dekar)")
ax.tick_params(axis="x", rotation=30)

# Add mean markers
means = wheat.groupby("bolge")["verim_kg_dekar"].mean().reindex(region_order)
for i, m in enumerate(means.values):
    ax.plot(i, m, "D", color="red", markersize=8, zorder=5)
ax.plot([], [], "D", color="red", markersize=8, label="Bölge Ortalaması")
ax.legend()

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/10_anova_regions.png")
plt.close()
print("  → Grafik kaydedildi: 10_anova_regions.png")

# ----------------------------------------------------------
# TEST 3: Chi-Square Test of Independence
# Question: Is there a significant association between geographical
# region and yield class (high/low)?
# ----------------------------------------------------------
print("\n" + "-" * 60)
print("TEST 3: Ki-Kare Bağımsızlık Testi (Chi-Square)")
print("-" * 60)
print("Soru: Coğrafi bölge ile verim sınıfı (Yüksek/Düşük) arasında")
print("      istatistiksel olarak anlamlı bir ilişki var mıdır?")
print()

contingency = pd.crosstab(df["bolge"], df["verim_sinifi_label"])
chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency)

print(f"  H₀: Bölge ve verim sınıfı birbirinden bağımsızdır")
print(f"  H₁: Bölge ve verim sınıfı arasında anlamlı bir ilişki vardır")
print(f"")
print(f"  Çapraz tablo (gözlenen):")
print(f"  {contingency.to_string()}")
print(f"")
print(f"  Ki-kare istatistiği: χ² = {chi2:.4f}")
print(f"  Serbestlik derecesi: df = {dof}")
print(f"  p-değeri: p = {p_value_chi:.2e}")
print(f"  Anlamlılık düzeyi: α = 0.05")
print(f"")
if p_value_chi < 0.05:
    print(f"  ✓ SONUÇ: p < 0.05 → H₀ REDDEDİLDİ.")
    print(f"    Coğrafi bölge ile verim sınıfı arasında istatistiksel")
    print(f"    olarak anlamlı bir ilişki bulunmaktadır.")
else:
    print(f"  ✗ SONUÇ: p ≥ 0.05 → H₀ reddedilemez.")

# Cramér's V effect size
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
print(f"  Etki büyüklüğü (Cramér's V): {cramers_v:.4f}")

# Visualization for Test 3
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Stacked bar chart
proportions = contingency.div(contingency.sum(axis=1), axis=0)
proportions = proportions.reindex(region_order)
proportions.plot(kind="bar", stacked=True, ax=axes[0], color=["#e74c3c", "#2ecc71"], edgecolor="black")
axes[0].set_title(f"Ki-Kare Testi: χ²={chi2:.1f}, p={p_value_chi:.2e}")
axes[0].set_xlabel("Coğrafi Bölge")
axes[0].set_ylabel("Oran")
axes[0].legend(title="Verim Sınıfı")
axes[0].tick_params(axis="x", rotation=45)
axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

# Heatmap of observed vs expected
diff = contingency.values - expected
diff_df = pd.DataFrame(diff, index=contingency.index, columns=contingency.columns)
diff_df = diff_df.reindex(region_order)
sns.heatmap(diff_df, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
            ax=axes[1], linewidths=0.5)
axes[1].set_title("Gözlenen − Beklenen Farklar")
axes[1].set_xlabel("Verim Sınıfı")
axes[1].set_ylabel("Bölge")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/11_chi_square_region_yield.png")
plt.close()
print("  → Grafik kaydedildi: 11_chi_square_region_yield.png")

# ----------------------------------------------------------
# TEST 4 (BONUS): Pearson Correlation Test
# Question: Is there a significant linear correlation between
# temperature and wheat yield?
# ----------------------------------------------------------
print("\n" + "-" * 60)
print("TEST 4: Pearson Korelasyon Testi")
print("-" * 60)
print("Soru: Ortalama sıcaklık ile buğday verimi arasında")
print("      istatistiksel olarak anlamlı bir doğrusal ilişki var mıdır?")
print()

r, p_value_corr = stats.pearsonr(wheat["ort_sicaklik_C"], wheat["verim_kg_dekar"])

print(f"  H₀: ρ = 0 (Sıcaklık ile verim arasında korelasyon yoktur)")
print(f"  H₁: ρ ≠ 0 (Korelasyon vardır)")
print(f"")
print(f"  Pearson r: {r:.4f}")
print(f"  p-değeri: p = {p_value_corr:.2e}")
print(f"")
if p_value_corr < 0.05:
    direction = "pozitif" if r > 0 else "negatif"
    strength = "zayıf" if abs(r) < 0.3 else ("orta" if abs(r) < 0.6 else "güçlü")
    print(f"  ✓ SONUÇ: p < 0.05 → H₀ REDDEDİLDİ.")
    print(f"    Sıcaklık ile buğday verimi arasında {strength} düzeyde")
    print(f"    {direction} bir korelasyon bulunmaktadır (r={r:.4f}).")
else:
    print(f"  ✗ SONUÇ: p ≥ 0.05 → H₀ reddedilemez.")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ÖZET")
print("=" * 70)
print(f"""
Veri seti: {len(df)} gözlem, {df['il'].nunique()} il, {len(crops)} ürün, {df['yil'].nunique()} yıl ({df['yil'].min()}-{df['yil'].max()})
Özellik sayısı: {len(df.columns)} (türetilmiş özellikler dahil)
Hedef değişken: verim_sinifi (0=Düşük, 1=Yüksek — medyan bazlı)

Hipotez Testi Sonuçları:
  Test 1 (t-testi)  : Yağış grupları arasında verim farkı → p={p_value:.6f}
  Test 2 (ANOVA)    : Bölgeler arasında verim farkı        → p={p_value_anova:.2e}
  Test 3 (Ki-kare)  : Bölge-verim sınıfı ilişkisi          → p={p_value_chi:.2e}
  Test 4 (Pearson)  : Sıcaklık-verim korelasyonu            → r={r:.4f}, p={p_value_corr:.2e}

Toplam {len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])} görselleştirme figures/ klasörüne kaydedildi.

Sonraki adım (5 Mayıs): ML modelleri (Logistic Regression, Random Forest, XGBoost)
""")
print("Analiz tamamlandı.")
