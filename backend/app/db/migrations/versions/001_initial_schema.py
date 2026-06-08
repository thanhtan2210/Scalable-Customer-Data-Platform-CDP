"""Initial schema setup - create audit and customer tables.

Revision ID: 001
Revises:
Create Date: 2026-01-28 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema with audit and customer tables."""
    # Create audit_log table for tracking data changes
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("table_name", sa.String(255), nullable=False),
        # INSERT, UPDATE, DELETE
        sa.Column("operation", sa.String(50), nullable=False),
        sa.Column("record_id", sa.String(255), nullable=True),
        # JSON or text describing what changed
        sa.Column("changes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("user_id", sa.String(255), nullable=True),
    )
    op.create_index(
        "ix_audit_log_table_created", "audit_log", ["table_name", "created_at"]
    )

    # Create customer_metadata table for storing customer feature metadata
    op.create_table(
        "customer_metadata",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("customer_id", sa.String(255), nullable=False, unique=True),
        sa.Column("total_purchase_amount", sa.Float, default=0.0),
        sa.Column("total_purchases", sa.Integer, default=0),
        sa.Column("last_purchase_date", sa.DateTime, nullable=True),
        sa.Column("churn_probability", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_customer_metadata_customer_id", "customer_metadata", ["customer_id"]
    )


def downgrade() -> None:
    """Drop tables in reverse order."""
    op.drop_index("ix_customer_metadata_customer_id", "customer_metadata")
    op.drop_table("customer_metadata")

    op.drop_index("ix_audit_log_table_created", "audit_log")
    op.drop_table("audit_log")
