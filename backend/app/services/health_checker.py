"""Background health checker service."""

import asyncio
from datetime import datetime
from typing import Optional

from sqlmodel import select

from app.database import get_session
from app.models import RunningInstance, ServerProfile
from app.services.server_manager import server_manager


class HealthChecker:
    """Background service that monitors server health."""

    def __init__(self, interval: int = 30):
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the health check loop."""
        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())

    async def stop(self):
        """Stop the health check loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                await self._check_all_servers()
            except Exception as e:
                print(f"Health check error: {e}")

            await asyncio.sleep(self.interval)

    async def _check_all_servers(self):
        """Check health of all running servers."""
        async with get_session() as session:
            # Get all running instances
            stmt = select(RunningInstance)
            result = await session.execute(stmt)
            instances = result.scalars().all()

            for instance in instances:
                # Get profile
                profile_stmt = select(ServerProfile).where(
                    ServerProfile.id == instance.profile_id
                )
                profile_result = await session.execute(profile_stmt)
                profile = profile_result.scalar_one_or_none()

                if not profile:
                    continue

                # Check if process is still running
                if not server_manager.is_running(instance.profile_id):
                    instance.health_status = "stopped"
                    instance.last_health_check = datetime.utcnow()
                    session.add(instance)
                    continue

                # Check health endpoint
                health = await server_manager.check_health(profile)

                # Update instance
                instance.health_status = health["status"]
                instance.last_health_check = datetime.utcnow()
                session.add(instance)

            await session.commit()


# Singleton instance
health_checker = HealthChecker()
