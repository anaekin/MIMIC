import random

def get_templates(class_name, n_samples=None):
    """
    Returns a list of ~1000 diverse templates for a given class name.
    Combinatorially generated to cover lighting, pose, style, and context variations.
    """
    
    # 1. Base Media Types (30)
    media_types = [
        "a photo of a {}",
        "a photograph of a {}",
        "an image of a {}",
        "a picture of a {}",
        "a shot of a {}",
        "a snap of a {}",
        "a polaroid of a {}",
        "a daguerreotype of a {}",
        "a black and white photo of a {}",
        "a painting of a {}",
        "a drawing of a {}",
        "a sketch of a {}",
        "an oil painting of a {}",
        "a watercolor painting of a {}",
        "a cartoon of a {}",
        "a rendering of a {}",
        "a 3D rendering of a {}",
        "a sculpture of a {}",
        "graffiti of a {}",
        "a tattoo of a {}",
        "a sticker of a {}",
        "an illustration of a {}",
        "concept art of a {}",
        "a poster of a {}",
        "pixel art of a {}",
        "ASCII art of a {}",
        "a silhouette of a {}",
        "a stencil of a {}",
        "a mosaic of a {}",
        "embroidery of a {}",
    ]

    # 2. Adjective Modifiers (25)
    adjectives = [
        "small", "large", "giant", "tiny", "miniature",
        "clean", "dirty", "muddy", "wet", "dry",
        "colorful", "shiny", "bright", "dark", "dim",
        "blurry", "sharp", "focused", "pixelated", "grainy",
        "old", "new", "vintage", "futuristic", "broken",
    ]

    # 3. Context/Background (20)
    contexts = [
        "in the dark", "in the rain", "in the snow", "in the fog",
        "in the sun", "on the grass", "on the floor", "on a table",
        "on the beach", "in the water", "underwater", "in the sky",
        "in a forest", "in a jungle", "in a desert", "in a city",
        "in a room", "background", "close-up", "far away",
    ]

    # 4. Stylistic/Quality modifiers (15)
    qualities = [
        "high quality", "low quality", "4k", "8k", "HD",
        "masterpiece", "trending on artstation", "award winning",
        "photorealistic", "surreal", "abstract", "minimalist",
        "detailed", "intricate", "rough",
    ]

    templates = []

    # Strategy 1: Simple Media + Class (Media Types) -> ~30
    for media in media_types:
        templates.append(media.format(class_name))

    # Strategy 2: Media + Adjective + Class -> 30 * 25 = 750 (sampled or full)
    # "a photo of a large {}"
    # We need to be careful with "a/an" but for simplicity we rely on the template's 'a'.
    for media in media_types:
        for adj in adjectives:
            # insertion: "a photo of a {}" -> "a photo of a {adj} {class}"
            # This is tricky with string formatting. 
            # Simplified: Construct strings manually
            prefix = media.split("{}")[0] # "a photo of a "
            templates.append(f"{prefix}{adj} {class_name}")

    # Strategy 3: Media + Class + Context -> 30 * 20 = 600
    for media in media_types:
        for ctx in contexts:
            templates.append(f"{media.format(class_name)} {ctx}")

    # Strategy 4: Quality + Media + Class -> 15 * 30 = 450
    for qual in qualities:
        for media in media_types:
            templates.append(f"{qual} {media.format(class_name)}")

    # Strategy 5: Random Mixes (Specific curated ones) -> ~50
    curated = [
        f"a photo of a {class_name}",
        f"a good photo of a {class_name}",
        f"a cropped photo of the {class_name}",
        f"the {class_name} is looking at the camera",
        f"a {class_name} in a video game",
        f"a {class_name} in a movie",
        f"art of the {class_name}",
        f"a webcam photo of a {class_name}",
        f"a security camera footage of a {class_name}",
        f"many {class_name}s",
        f"a single {class_name}",
        f"two {class_name}s",
        f"part of a {class_name}",
        f"the side of a {class_name}",
        f"the back of a {class_name}",
        f"a top view of a {class_name}",
        f"a bottom view of a {class_name}",
        f"rotated {class_name}",
        f"inverted {class_name}",
    ]
    templates.extend(curated)

    # Dedup and sort
    templates = sorted(list(set(templates)))
    
    if n_samples is not None:
        if len(templates) > n_samples:
            return random.sample(templates, n_samples)
        else:
            # If we requested more than we have, we might concatenate or repeat
            # to match exactly n_samples
            factor = (n_samples // len(templates)) + 1
            templates = templates * factor
            return templates[:n_samples]
    
    return templates
