import pandas as pd

path = r"C:\Users\KarriBhavya\PycharmProjects\Gradient_cross_Slope_geometry\Data_1\geometry.csv"

df = pd.read_csv(path, header=None, names=[
    "ID","networkID","sectionName","locFrom","locTo","lane",
    "measDate","gradient","crossfall","RAMMID","latitude","longitude"
])

# Convert numeric columns
df["gradient"] = pd.to_numeric(df["gradient"], errors="coerce")
df["crossfall"] = pd.to_numeric(df["crossfall"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# GROUP BY segment columns
segments = df.groupby(
    ["networkID","sectionName","locFrom","locTo","lane"]
).agg({
    "gradient": "mean",
    "crossfall": "mean",
    "latitude": "mean",
    "longitude": "mean"
}).reset_index()

print("Unique segments:", len(segments))
segments.to_csv("geometry_segments.csv", index=False)
