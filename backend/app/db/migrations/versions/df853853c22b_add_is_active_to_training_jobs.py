"""add is_active to training_jobs

Revision ID: df853853c22b
Revises: 001
Create Date: 2026-07-07 14:54:50.583627

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "df853853c22b"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add column as nullable first, then populate and set default/not-null if needed
    op.add_column(
        "training_jobs",
        sa.Column(
            "is_active", sa.Boolean(), nullable=True, server_default=sa.text("false")
        ),
    )
    # Set default values for any existing records
    op.execute("UPDATE training_jobs SET is_active = false WHERE is_active IS NULL")


def downgrade() -> None:
    op.drop_column("training_jobs", "is_active")
