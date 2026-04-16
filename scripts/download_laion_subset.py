from bing_image_downloader import downloader

print("Downloading LAION-like dataset (paper-style)...")

# Each cluster simulates duplicated captions
clusters = [
    #  Faces (VERY HIGH duplication in LAION)
    [
        "stock photo smiling woman portrait",
        "portrait smiling woman studio lighting",
        "professional headshot smiling woman",
        "close up face smiling woman",
        "studio portrait female face"
    ],

    #  Headshots
    [
        "professional headshot man suit",
        "portrait man suit studio lighting",
        "corporate headshot male",
        "businessman portrait studio",
        "linkedin profile photo man suit"
    ],

    #  Products
    [
        "product photo white sneakers",
        "nike white shoes product image",
        "white sneakers isolated background",
        "studio shot white shoes",
        "shoe catalog image white sneakers"
    ],

    #  Generic stock images
    [
        "stock photo laptop on desk coffee",
        "workspace laptop coffee cup desk",
        "office desk laptop aesthetic",
        "minimal desk laptop coffee",
        "workspace setup laptop top view"
    ],

    #  Animals (also duplicated often)
    [
        "dog sitting on grass photo",
        "cute dog outdoor grass portrait",
        "pet dog sitting lawn photo",
        "dog portrait natural light",
        "dog looking camera outdoor"
    ]
]

# Download images
for cluster in clusters:
    for query in cluster:
        downloader.download(
            query,
            limit=200,              # 200 × 5 × 5 ≈ 5000 images
            output_dir="laion_subset",
            adult_filter_off=True,
            force_replace=False
        )

print("Download complete!")

