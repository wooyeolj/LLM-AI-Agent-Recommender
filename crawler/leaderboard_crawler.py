from typing import Dict


class LeaderboardCrawler:
    async def load_chatbot_arena(self) -> Dict[str, float]:
        # Chatbot Arena는 공개 API 미제공 (gated dataset, JS SPA)
        return {}

    def get_elo(self, arena_data: Dict[str, float], model_id: str) -> float:
        return -1.0


leaderboard_crawler = LeaderboardCrawler()
