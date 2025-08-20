# generated-adventures
A Python app for generating short D&amp;D like adventures with generated story, voice narration and imagery using Multimodal LLMs.

### How to Run
1. ``uv run uvicorn webapp.main:app --reload`` for playing the game.
2. ``SHOW_DM_NOTES=1 uv run uvicorn webapp.main:app --reload`` for playing the game with DM notes visible.

### App Flow
1. User is prompted to enter a how many players will be playing the adventure.
2. User is prompted to select out of 3 generated fantasy adventure scenarios.
3. The story of the adventure is generated including the setting, plot, main quest, important NPCs and locations.
4. User is prompted to select a character for each player out of 6 generated characters.
5. Party overview is displayed with selected characters.
6. The opening scene is generated and narrated.
7. Image is generated to represent the opening scene.
8. One of the characters is prompted for an action.
9. The action is processed and the next scene is generated and narrated.
10. The process continues until the adventure is completed.
