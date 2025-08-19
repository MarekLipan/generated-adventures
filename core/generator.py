"""
Handles content generation via external APIs.
"""


def generate_scenarios() -> list[str]:
    """Calls an external service to generate scenarios."""
    # In a real app, you would call Gemini here
    return [
        "Rescue in Frostvale",
        "Siege of Emberhold",
        "The Lost Tomb of Varaxis",
    ]


def generate_characters() -> list[str]:
    """Calls an external service to generate characters."""
    # In a real app, you would call Gemini here
    return ["Thalion the Ranger", "Mira the Mage", "Gruk the Barbarian"]


def generate_scenario_details(scenario_name: str) -> str:
    """Generates the detailed story for the selected scenario."""
    # In a real app, you would make a detailed call to Gemini or another model here.
    # This placeholder returns markdown-formatted story content.
    return f"""
# {scenario_name}

## The Setting
A realm shrouded in an unnatural, perpetual twilight. The towering peaks of the Dragon's Tooth mountains loom over the land, their shadows hiding ancient secrets and forgotten paths. The once-vibrant forests are now twisted and silent, and the rivers run sluggishly with dark, cold water.

## The Plot
The legendary Sunstone, the only source of light and warmth for the kingdom of Eldoria, has been stolen by the shadowy Sorcerer Malakor. Without it, the land is succumbing to a creeping, life-draining cold. Malakor plans to use the Sunstone's power to plunge the world into eternal darkness and rule over a kingdom of shadows.

## Main Quest
The heroes must journey to Malakor's fortress, the Obsidian Spire, which lies deep within the treacherous Shadowfen Marshes. They must navigate the dangers of the corrupted land, overcome Malakor's minions, and retrieve the Sunstone before the last light in Eldoria is extinguished forever.

## Important NPCs
- **Elara, the Seer**: An old, blind woman who can guide the party with her visions, but her help comes at a price.
- **Garrick, the Disgraced Knight**: A former royal guard exiled for a crime he did not commit. He seeks redemption and knows a secret path to the Obsidian Spire.

## Key Locations
- **The Whispering Village**: A settlement on the edge of the Shadowfen Marshes, where the locals are paralyzed by fear and suspicion.
- **The Sunken Temple**: An ancient ruin submerged in the marsh, holding a clue to Malakor's weakness.
- **The Obsidian Spire**: Malakor's dark fortress, a twisted tower of black glass that seems to absorb the very light around it.
"""
