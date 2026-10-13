"""Persist inbox summaries and weekly digest delivery state."""

from alembic import op
import sqlalchemy as sa

revision = "a4c9d0e1f2a3"
down_revision = "86d6b38a5c75"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("user") as batch:
        batch.add_column(sa.Column("sync_alerts", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("weekly_digest_last_sent_week", sa.String(), nullable=True))


def downgrade():
    with op.batch_alter_table("user") as batch:
        batch.drop_column("weekly_digest_last_sent_week")
        batch.drop_column("sync_alerts")
