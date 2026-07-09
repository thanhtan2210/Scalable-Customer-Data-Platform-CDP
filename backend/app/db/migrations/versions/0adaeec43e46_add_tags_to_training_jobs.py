"""add tags to training_jobs

Revision ID: 0adaeec43e46
Revises: ece5f56dca64
Create Date: 2026-07-09 09:16:34.455695

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0adaeec43e46'
down_revision = 'ece5f56dca64'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('training_jobs', sa.Column('tags', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('training_jobs', 'tags')
