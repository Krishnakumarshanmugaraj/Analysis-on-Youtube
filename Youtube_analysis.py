import pandas as pd
import numpy as np
import json
import re
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

PALETTE = {
    "saffron":   "#FF9933",
    "deep_blue": "#0F4C81",
    "crimson":   "#C8102E",
    "teal":      "#138D90",
    "gold":      "#D4AF37",
    "charcoal":  "#2B2D42",
}

ENT_CATS  = ["Entertainment", "Music", "Comedy", "Film & Animation", "People & Blogs"]
EDU_CATS  = ["Education", "Science & Technology", "Howto & Style"]
OTHER_CATS = ["News & Politics"]
FOCUS_CATS = ENT_CATS + EDU_CATS + OTHER_CATS

def cat_color(c):
    if c in ENT_CATS:  return PALETTE["crimson"]
    if c in EDU_CATS:  return PALETTE["deep_blue"]
    return PALETTE["charcoal"]

print("Loading data...")
df = pd.read_csv("INvideos.csv", encoding="utf-8", on_bad_lines="skip")

with open("IN_category_id.json") as f:
    cat_json = json.load(f)

cat_map = {int(item["id"]): item["snippet"]["title"] for item in cat_json["items"]}
df["category"] = df["category_id"].map(cat_map)

df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m", errors="coerce")
df["publish_time"]  = pd.to_datetime(df["publish_time"], utc=True, errors="coerce")
df["publish_time_naive"] = df["publish_time"].dt.tz_localize(None)

df["hours_to_trend"] = (
    df["trending_date"] - df["publish_time_naive"]
).dt.total_seconds() / 3600
df["days_to_trend"] = df["hours_to_trend"] / 24
df["publish_hour"]  = df["publish_time"].dt.hour
df["publish_day"]   = df["publish_time"].dt.day_name()

df = df[(df["hours_to_trend"] > 0) & (df["hours_to_trend"] < 24 * 60)]
df["engagement_per_1k"] = (df["likes"] + df["comment_count"]) / df["views"] * 1000
df["like_ratio"] = df["likes"] / (df["likes"] + df["dislikes"] + 1)
df["tag_count"]  = df["tags"].apply(lambda x: 0 if str(x) == "[none]" else len(str(x).split("|")))

print(f"Records      : {len(df):,}")
print(f"Unique videos: {df['video_id'].nunique():,}")
print(f"Date range   : {df['trending_date'].min().date()} → {df['trending_date'].max().date()}")
print(f"Categories   : {df['category'].nunique()}")


shelf_life = (
    df.groupby(["video_id", "category"])["trending_date"]
    .nunique().reset_index()
    .rename(columns={"trending_date": "trend_days"})
)


print("\nGenerating Chart 1 — Category View Gap...")
cat_views = (
    df[df["category"].isin(FOCUS_CATS)]
    .groupby("category")["views"]
    .agg(mean="mean", median="median", count="count")
    .sort_values("mean", ascending=False).reset_index()
)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")
colors = [cat_color(c) for c in cat_views["category"]]
bars = ax.barh(cat_views["category"], cat_views["mean"] / 1e6,
               color=colors, edgecolor="white", height=0.6)
ax.invert_yaxis()
ax.set_xlabel("Average Views per Trending Video (Millions)", fontsize=11)
ax.set_title("The View Gap: Entertainment vs Education on Indian YouTube",
             fontsize=14, fontweight="bold", pad=15)
for bar, val in zip(bars, cat_views["mean"]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val/1e6:.2f}M", va="center", fontsize=10, fontweight="bold")
red_p  = mpatches.Patch(color=PALETTE["crimson"],   label="Entertainment")
blue_p = mpatches.Patch(color=PALETTE["deep_blue"], label="Learning / Tech")
gray_p = mpatches.Patch(color=PALETTE["charcoal"],  label="Other")
ax.legend(handles=[red_p, blue_p, gray_p], fontsize=10, frameon=False)
ax.text(0.5, -0.14, "Source: Kaggle INvideos.csv — India Trending Dataset",
        transform=ax.transAxes, ha="center", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig("chart1_category_views.png", dpi=150, facecolor="#FAFAFA")
plt.close()
print("  Saved chart1_category_views.png")


print("Generating Chart 2 — Engagement Quality...")
eng_quality = (
    df[df["category"].isin(FOCUS_CATS)]
    .groupby("category")["engagement_per_1k"]
    .median().sort_values(ascending=False).reset_index()
)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")
colors2 = [cat_color(c) for c in eng_quality["category"]]
bars = ax.barh(eng_quality["category"], eng_quality["engagement_per_1k"],
               color=colors2, edgecolor="white", height=0.6)
ax.invert_yaxis()
ax.set_xlabel("Median (Likes + Comments) per 1,000 Views", fontsize=11)
ax.set_title("Engagement Quality: Does Learning Content Get More Meaningful Interaction?",
             fontsize=14, fontweight="bold", pad=15)
for bar, val in zip(bars, eng_quality["engagement_per_1k"]):
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=10, fontweight="bold")
ax.legend(handles=[red_p, blue_p, gray_p], fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig("chart2_engagement_quality.png", dpi=150, facecolor="#FAFAFA")
plt.close()
print("  Saved chart2_engagement_quality.png")


print("Generating Chart 3 — View Velocity...")
velocity = (
    df[df["category"].isin(FOCUS_CATS)]
    .dropna(subset=["hours_to_trend"])
    .groupby("category")["hours_to_trend"]
    .median().sort_values().reset_index()
)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")
colors3 = [cat_color(c) for c in velocity["category"]]
bars = ax.barh(velocity["category"], velocity["hours_to_trend"],
               color=colors3, edgecolor="white", height=0.6)
ax.set_xlabel("Median Hours from Upload → Trending", fontsize=11)
ax.set_title("View Velocity: How Fast Does Content Catch Fire?",
             fontsize=14, fontweight="bold", pad=15)
for bar, val in zip(bars, velocity["hours_to_trend"]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} hrs", va="center", fontsize=10, fontweight="bold")
ax.legend(handles=[red_p, blue_p, gray_p], fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig("chart3_view_velocity.png", dpi=150, facecolor="#FAFAFA")
plt.close()
print("  Saved chart3_view_velocity.png")


print("Generating Chart 4 — Title Sentiment...")
EMOTIONAL_KEYWORDS = [
    "shocking", "amazing", "unbelievable", "must watch", "viral", "exposed",
    "truth", "secret", "real story", "reaction", "reacts", "best", "worst",
    "epic", "funny", "prank", "challenge", "vs", "wins", "fails", "craziest",
    "mind blowing", "leaked", "controversy", "emotional", "revealed",
    "நடந்தது", "உண்மை", "அதிர்ச்சி",
]
INFO_KEYWORDS = [
    "tutorial", "how to", "learn", "course", "lesson", "introduction",
    "basics", "step by step", "lecture", "chapter", "full course",
    "explained", "beginners", "guide", "training", "concept",
    "python", "java", "data science", "machine learning", "ai",
]

def classify_title(title):
    t = str(title).lower()
    if any(k in t for k in EMOTIONAL_KEYWORDS): return "Emotional / Curiosity"
    if any(k in t for k in INFO_KEYWORDS):       return "Informational"
    if "?" in t or "!" in t:                      return "Emotional / Curiosity"
    return "Neutral"

df["title_type"] = df["title"].apply(classify_title)

title_stats = (
    df.groupby("title_type")
    .agg(avg_views=("views","mean"), count=("video_id","count"))
    .reset_index().sort_values("avg_views", ascending=False)
)

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")
color4_map = {
    "Emotional / Curiosity": PALETTE["crimson"],
    "Informational":         PALETTE["deep_blue"],
    "Neutral":               PALETTE["charcoal"],
}
bars = ax.bar(title_stats["title_type"],
              title_stats["avg_views"] / 1e6,
              color=[color4_map[t] for t in title_stats["title_type"]],
              width=0.5, edgecolor="white")
ax.set_ylabel("Average Views (Millions)", fontsize=11)
ax.set_title("Title Type vs Average Views — Indian YouTube Trending",
             fontsize=14, fontweight="bold", pad=15)
for bar, val, n in zip(bars, title_stats["avg_views"], title_stats["count"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val/1e6:.2f}M\n(n={n:,})", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("chart4_title_sentiment.png", dpi=150, facecolor="#FAFAFA")
plt.close()
print("  Saved chart4_title_sentiment.png")

print("Generating Chart 5 — Archetype Channel Comparison...")
hand_picked = pd.DataFrame({
    "channel": [
        "Lifestyle Vlogger 1\n(Tamil, since 2023)",
        "Travel/Bike Vlogger\n(Tamil, since 2019)",
        "Tech Educator\n(Tamil, since 2015)",
        "AI/Learning Creator\n(English, since 2023)",
        "Business Educator\n(Tamil, since 2021)",
    ],
    "type": [
        "Vlog/Emotional", "Travel/Bike Vlog", "Tech (Tamil)",
        "AI/Learning", "Business/Learning",
    ],
    "subscribers_m": [5.9,  4.94, 1.89, 0.15, 0.35],
    "avg_views_k":   [2500, 750,  100,  30,   50  ],
})

arch_colors = [
    PALETTE["crimson"] if t in ["Vlog/Emotional", "Travel/Bike Vlog"]
    else PALETTE["deep_blue"]
    for t in hand_picked["type"]
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("#FAFAFA")

bars1 = axes[0].barh(hand_picked["channel"], hand_picked["subscribers_m"],
                     color=arch_colors, edgecolor="white", height=0.55)
axes[0].invert_yaxis()
axes[0].set_xlabel("Subscribers (Millions)", fontsize=11)
axes[0].set_title("Subscriber Base", fontweight="bold", fontsize=12)
axes[0].set_facecolor("#FAFAFA")
for bar, val in zip(bars1, hand_picked["subscribers_m"]):
    axes[0].text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}M", va="center", fontsize=10, fontweight="bold")

bars2 = axes[1].barh(hand_picked["channel"], hand_picked["avg_views_k"],
                     color=arch_colors, edgecolor="white", height=0.55)
axes[1].invert_yaxis()
axes[1].set_xlabel("Avg Views per Video", fontsize=11)
axes[1].set_title("Avg Views per Video", fontweight="bold", fontsize=12)
axes[1].set_facecolor("#FAFAFA")
for bar, val in zip(bars2, hand_picked["avg_views_k"]):
    label = f"{val/1000:.1f}M" if val >= 1000 else f"{val}K"
    axes[1].text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=10, fontweight="bold")

red_patch  = mpatches.Patch(color=PALETTE["crimson"],   label="Entertainment / Vlog")
blue_patch = mpatches.Patch(color=PALETTE["deep_blue"], label="Learning / Tech")
fig.legend(handles=[red_patch, blue_patch], loc="lower center",
           ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("The Tamil YouTube View Gap: 5 Archetypes, 2 Realities",
             fontsize=15, fontweight="bold", y=1.02)
fig.text(0.5, -0.1,
         "Source: Public channel stats (June 2026)  •  Archetype labels used — no individual creators named",
         ha="center", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig("chart5_archetype_channels.png", dpi=150,
            bbox_inches="tight", facecolor="#FAFAFA")
plt.close()
print("  Saved chart5_archetype_channels.png")


print("Generating Chart 6 — Trend Shelf Life...")
shelf_by_cat = (
    shelf_life[shelf_life["category"].isin(FOCUS_CATS)]
    .groupby("category")["trend_days"]
    .median().sort_values(ascending=False).reset_index()
)

fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")
colors6 = [cat_color(c) for c in shelf_by_cat["category"]]
bars = ax.barh(shelf_by_cat["category"], shelf_by_cat["trend_days"],
               color=colors6, edgecolor="white", height=0.6)
ax.invert_yaxis()
ax.set_xlabel("Median Days on Trending List", fontsize=11)
ax.set_title("Trend Shelf Life: How Long Does Virality Last?",
             fontsize=14, fontweight="bold", pad=15)
for bar, val in zip(bars, shelf_by_cat["trend_days"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} days", va="center", fontsize=10, fontweight="bold")
ax.legend(handles=[red_p, blue_p, gray_p], fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig("chart6_shelf_life.png", dpi=150, facecolor="#FAFAFA")
plt.close()
print("  Saved chart6_shelf_life.png")

edu_text = " ".join(df[df["category"] == "Education"]["title"].dropna().astype(str).str.lower())
ent_text = " ".join(df[df["category"] == "Entertainment"]["title"].dropna().astype(str).str.lower())

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("#FAFAFA")
if edu_text.strip():
    axes[0].imshow(WordCloud(width=800, height=400, background_color="white",
                             colormap="Blues", max_words=80).generate(edu_text))
axes[0].axis("off")
axes[0].set_title("Words in Education Trending Titles", fontweight="bold", fontsize=13)
if ent_text.strip():
    axes[1].imshow(WordCloud(width=800, height=400, background_color="white",
                             colormap="Reds", max_words=80).generate(ent_text))
axes[1].axis("off")
axes[1].set_title("Words in Entertainment Trending Titles", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("chart7_wordcloud.png", dpi=150, facecolor="#FAFAFA")
plt.close()
print("  Saved chart7_wordcloud.png")

print("Generating Chart 8 — Best Day to Post...")
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_views = (
    df[df["category"].isin(ENT_CATS + EDU_CATS)]
    .groupby(["publish_day", "category"])["views"]
    .mean().reset_index()
)
day_views["publish_day"] = pd.Categorical(day_views["publish_day"],
                                           categories=day_order, ordered=True)
day_views = day_views.sort_values("publish_day")

fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")
for cat, grp in day_views.groupby("category"):
    color = PALETTE["crimson"] if cat in ENT_CATS else PALETTE["deep_blue"]
    label = "Entertainment" if cat in ENT_CATS else "Learning/Tech"
    ax.plot(grp["publish_day"], grp["views"] / 1e6,
            color=color, linewidth=2.5, label=label, alpha=0.85)
ax.set_xlabel("Day of Upload", fontsize=11)
ax.set_ylabel("Avg Views (Millions)", fontsize=11)
ax.set_title("Does Upload Day Affect Views?", fontsize=14, fontweight="bold", pad=15)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig("chart8_best_day.png", dpi=150, facecolor="#FAFAFA")
plt.close()
print("  Saved chart8_best_day.png")

ent_avg = df[df["category"] == "Entertainment"]["views"].mean()
edu_avg = df[df["category"] == "Education"]["views"].mean()
ent_eng = df[df["category"] == "Entertainment"]["engagement_per_1k"].median()
edu_eng = df[df["category"] == "Education"]["engagement_per_1k"].median()
ent_hrs = df[df["category"] == "Entertainment"]["hours_to_trend"].median()
edu_hrs = df[df["category"] == "Education"]["hours_to_trend"].median()

print("\n" + "="*50)
print("="*50)
print(f"Entertainment avg views  : {ent_avg/1e6:.2f}M")
print(f"Education avg views      : {edu_avg/1e6:.2f}M")
print(f"View gap ratio           : {ent_avg/edu_avg:.1f}x")
print(f"Entertainment engagement : {ent_eng:.1f} per 1K views")
print(f"Education engagement     : {edu_eng:.1f} per 1K views")
print(f"Entertainment → trending : {ent_hrs:.0f} hours")
print(f"Education → trending     : {edu_hrs:.0f} hours")
print("="*50)