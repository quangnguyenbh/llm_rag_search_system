"""Role-based access control."""

from enum import Enum


class Role(str, Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


# Permission matrix
PERMISSIONS: dict[str, set[Role]] = {
    "query:read": {Role.ADMIN, Role.EDITOR, Role.VIEWER},
    "documents:read": {Role.ADMIN, Role.EDITOR, Role.VIEWER},
    "documents:write": {Role.ADMIN, Role.EDITOR},
    "documents:delete": {Role.ADMIN},
    "admin:read": {Role.ADMIN},
    "admin:write": {Role.ADMIN},
    "billing:read": {Role.ADMIN, Role.EDITOR},
    "billing:write": {Role.ADMIN},
}


def has_permission(role: Role, permission: str) -> bool:
    allowed_roles = PERMISSIONS.get(permission, set())
    return role in allowed_roles
