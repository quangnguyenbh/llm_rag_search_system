"""User repository — CRUD operations for the User model."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.postgres.models import Organization, User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: str) -> User | None:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def create(
        self,
        email: str,
        name: str,
        hashed_password: str,
        role: str = "viewer",
        organization_id: str | None = None,
    ) -> User:
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            name=name,
            hashed_password=hashed_password,
            role=role,
            organization_id=organization_id,
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def update_stripe_customer(self, user_id: str, stripe_customer_id: str) -> None:
        user = await self.get_by_id(user_id)
        if user:
            user.stripe_customer_id = stripe_customer_id
            await self.session.commit()

    async def count(self) -> int:
        from sqlalchemy import func, select

        result = await self.session.execute(select(func.count(User.id)))
        return result.scalar_one()


class OrganizationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, org_id: str) -> Organization | None:
        result = await self.session.execute(
            select(Organization).where(Organization.id == org_id)
        )
        return result.scalar_one_or_none()

    async def create(self, name: str, plan: str = "free") -> Organization:
        org = Organization(
            id=str(uuid.uuid4()),
            name=name,
            plan=plan,
        )
        self.session.add(org)
        await self.session.commit()
        await self.session.refresh(org)
        return org

    async def update_subscription(
        self, org_id: str, plan: str, stripe_subscription_id: str | None = None
    ) -> None:
        org = await self.get_by_id(org_id)
        if org:
            org.plan = plan
            org.stripe_subscription_id = stripe_subscription_id
            await self.session.commit()
