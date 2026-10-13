"""Preserve an unavailable Email model score as NULL, never a risk estimate."""

from alembic import op
import sqlalchemy as sa

revision = "c7e2b0a61f84"
down_revision = "a4c9d0e1f2a3"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("scanrecord") as batch:
        batch.alter_column("risk_score", existing_type=sa.Integer(), nullable=True)


def downgrade():
    # Reverting must not invent a score or delete review records to satisfy NOT NULL.
    if op.get_bind().execute(sa.text(
        "SELECT 1 FROM scanrecord WHERE risk_score IS NULL LIMIT 1"
    )).first() is not None:
        raise RuntimeError("Cannot restore NOT NULL while unavailable scan scores exist; preserve or explicitly resolve those records first")
    with op.batch_alter_table("scanrecord") as batch:
        batch.alter_column("risk_score", existing_type=sa.Integer(), nullable=False)
