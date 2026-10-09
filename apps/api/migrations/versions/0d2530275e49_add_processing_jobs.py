"""add processing jobs

Revision ID: 0d2530275e49
Revises: 0bc6a51efeff
Create Date: 2026-10-08 12:13:35.669798

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0d2530275e49'
down_revision: Union[str, Sequence[str], None] = '0bc6a51efeff'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'processing_jobs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('match_id', sa.String(length=36), nullable=False),
        sa.Column('video_id', sa.String(length=36), nullable=False),
        sa.Column('kind', sa.String(length=40), nullable=False),
        sa.Column('mode', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('progress', sa.Integer(), nullable=False),
        sa.Column('result_summary', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id']),
        sa.ForeignKeyConstraint(['video_id'], ['video_assets.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_processing_jobs_org_match', 'processing_jobs', ['organization_id', 'match_id'])
    op.create_index(op.f('ix_processing_jobs_organization_id'), 'processing_jobs', ['organization_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_processing_jobs_organization_id'), table_name='processing_jobs')
    op.drop_index('ix_processing_jobs_org_match', table_name='processing_jobs')
    op.drop_table('processing_jobs')
