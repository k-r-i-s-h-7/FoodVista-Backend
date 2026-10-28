FOOD_QUANTITY = {

    # ============================================================
    # AREA-BASED FOODS
    # estimated grams = segmented_area_cm2 * g_per_cm2
    # ============================================================

    "aloo_gobi": {
        "mode": "area",
        "g_per_cm2": 1.05,
    },

    "aloo_sabji": {
        "mode": "area",
        "g_per_cm2": 1.10,
    },

    "aloomasala": {
        "mode": "area",
        "g_per_cm2": 1.05,
    },

    "avial": {
        "mode": "area",
        "g_per_cm2": 1.10,
    },

    "bhindi": {
        "mode": "area",
        "g_per_cm2": 0.85,
    },

    "biryani": {
        "mode": "area",
        "g_per_cm2": 1.10,
    },

    "broccoli": {
        "mode": "area",
        "g_per_cm2": 0.65,
    },

    "cake": {
        "mode": "area",
        "g_per_cm2": 0.80,
    },

    "carrot": {
        "mode": "area",
        "g_per_cm2": 0.65,
    },

    "chicken_curry": {
        "mode": "area",
        "g_per_cm2": 1.15,
    },

    "chips": {
        "mode": "area",
        "g_per_cm2": 0.30,
    },

    "chole": {
        "mode": "area",
        "g_per_cm2": 1.15,
    },

    "chutney": {
        "mode": "area",
        "g_per_cm2": 0.90,
    },

    "dal": {
        "mode": "area",
        "g_per_cm2": 1.10,
    },

    "dry_fruits": {
        "mode": "area",
        "g_per_cm2": 0.70,
    },

    "dum_aloo": {
        "mode": "area",
        "g_per_cm2": 1.10,
    },

    "fish_curry": {
        "mode": "area",
        "g_per_cm2": 1.15,
    },

    "frenchfries": {
        "mode": "area",
        "g_per_cm2": 0.55,
    },

    "kheer": {
        "mode": "area",
        "g_per_cm2": 1.05,
    },

    "lasagna": {
        "mode": "area",
        "g_per_cm2": 1.05,
    },

    "mutton_curry": {
        "mode": "area",
        "g_per_cm2": 1.15,
    },

    "palak_paneer": {
        "mode": "area",
        "g_per_cm2": 1.10,
    },

    "pasta": {
        "mode": "area",
        "g_per_cm2": 0.95,
    },

    "poha": {
        "mode": "area",
        "g_per_cm2": 1.15,
    },

    "popcorn": {
        "mode": "area",
        "g_per_cm2": 0.25,
    },

    "poriyal": {
        "mode": "area",
        "g_per_cm2": 0.90,
    },

    "potato": {
        "mode": "area",
        "g_per_cm2": 1.00,
    },

    "rajma": {
        "mode": "area",
        "g_per_cm2": 1.15,
    },

    "ras_malai": {
        "mode": "area",
        "g_per_cm2": 1.00,
    },

    "rice": {
        "mode": "area",
        "g_per_cm2": 1.05,
    },

    "saag": {
        "mode": "area",
        "g_per_cm2": 0.85,
    },

    "salad": {
        "mode": "area",
        "g_per_cm2": 0.55,
    },

    "sambar": {
        "mode": "area",
        "g_per_cm2": 1.05,
    },

    "shahi_paneer": {
        "mode": "area",
        "g_per_cm2": 1.15,
    },


    # ============================================================
    # COUNT-BASED FOODS
    #
    # baseline_area_cm2 = projected area of one normal item
    # baseline_g        = mass of one normal item
    #
    # Actual grams:
    #
    # detected_area / baseline_area * baseline_g
    #
    # ============================================================

    "appam": {
        "mode": "count",
        "baseline_area_cm2": 110,
        "baseline_g": 55,
    },

    "bhatura": {
        "mode": "count",
        "baseline_area_cm2": 170,
        "baseline_g": 80,
    },

    "cookie": {
        "mode": "count",
        "baseline_area_cm2": 28,
        "baseline_g": 12,
    },

    "cracker": {
        "mode": "count",
        "baseline_area_cm2": 20,
        "baseline_g": 8,
    },

    "donuts": {
        "mode": "count",
        "baseline_area_cm2": 65,
        "baseline_g": 60,
    },

    "dosa": {
        "mode": "count",
        "baseline_area_cm2": 190,
        "baseline_g": 90,
    },

    "eggs": {
        "mode": "count",
        "baseline_area_cm2": 35,
        "baseline_g": 50,
    },

    "ghevar": {
        "mode": "count",
        "baseline_area_cm2": 95,
        "baseline_g": 70,
    },

    "gulab_jamun": {
        "mode": "count",
        "baseline_area_cm2": 28,
        "baseline_g": 25,
    },

    "idli": {
        "mode": "count",
        "baseline_area_cm2": 45,
        "baseline_g": 40,
    },

    "jalebi": {
        "mode": "count",
        "baseline_area_cm2": 40,
        "baseline_g": 30,
    },

    "kebab": {
        "mode": "count",
        "baseline_area_cm2": 55,
        "baseline_g": 50,
    },

    "kulfi": {
        "mode": "count",
        "baseline_area_cm2": 35,
        "baseline_g": 60,
    },

    "lemon": {
        "mode": "count",
        "baseline_area_cm2": 45,
        "baseline_g": 60,
    },

    "omelette": {
        "mode": "count",
        "baseline_area_cm2": 130,
        "baseline_g": 90,
    },

    "pakoda": {
        "mode": "count",
        "baseline_area_cm2": 35,
        "baseline_g": 30,
    },

    "pancake": {
        "mode": "count",
        "baseline_area_cm2": 100,
        "baseline_g": 55,
    },

    "roti": {
        "mode": "count",
        "baseline_area_cm2": 145,
        "baseline_g": 40,
    },

    "salatini": {
        "mode": "count",
        "baseline_area_cm2": 25,
        "baseline_g": 10,
    },

    "samosa": {
        "mode": "count",
        "baseline_area_cm2": 55,
        "baseline_g": 50,
    },

    "sandwich": {
        "mode": "count",
        "baseline_area_cm2": 150,
        "baseline_g": 150,
    },

    "tomato": {
        "mode": "count",
        "baseline_area_cm2": 45,
        "baseline_g": 100,
    },

    "uttapam": {
        "mode": "count",
        "baseline_area_cm2": 180,
        "baseline_g": 120,
    },

    "vada": {
        "mode": "count",
        "baseline_area_cm2": 45,
        "baseline_g": 60,
    },


    # ============================================================
    # LIQUIDS
    # ============================================================

    "chai": {
        "mode": "liquid",
        "ml_per_cm2": 0.65,
    },

    "lassi": {
        "mode": "liquid",
        "ml_per_cm2": 0.70,
    },

    "lemonade": {
        "mode": "liquid",
        "ml_per_cm2": 0.70,
    },
}