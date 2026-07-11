"""add indexes for production performance

Revision ID: ece5f56dca64
Revises: 8dd8a4bf1c14
Create Date: 2026-07-08 10:54:36.236816

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "ece5f56dca64"
down_revision = "8dd8a4bf1c14"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_training_jobs_dataset_id", "training_jobs", ["dataset_id"])
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"])
    op.create_index("ix_training_jobs_finished_at", "training_jobs", ["finished_at"])
    op.create_index("ix_datasets_status", "datasets", ["status"])


def downgrade() -> None:
    op.drop_index("ix_training_jobs_dataset_id", table_name="training_jobs")
    op.drop_index("ix_training_jobs_status", table_name="training_jobs")
    op.drop_index("ix_training_jobs_finished_at", table_name="training_jobs")
    op.drop_index("ix_datasets_status", table_name="datasets")
