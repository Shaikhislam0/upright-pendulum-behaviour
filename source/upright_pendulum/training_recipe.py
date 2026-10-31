"""Explicit cohort settings for the existing training variants; physics is unchanged."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingRecipe:
    gradient_clipping: str = "global"
    training_boundary_m: float | None = None
    boundary_failure_cost: float = 10.

    def __post_init__(self):
        if self.gradient_clipping not in ("global", "independent"):
            raise ValueError("gradient_clipping must be global or independent")
        if self.training_boundary_m is not None:
            value = self.training_boundary_m
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("training_boundary_m must be a finite positive number or None")
        value = self.boundary_failure_cost
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("boundary_failure_cost must be finite and positive")


def check_member_recipe(member, recipe):
    model = member.model
    if (getattr(model, "pendulum_gradient_clipping", "global") != recipe.gradient_clipping or
            getattr(model, "pendulum_training_boundary_m", None) != recipe.training_boundary_m or
            getattr(model, "pendulum_boundary_failure_cost", 10.) != recipe.boundary_failure_cost or
            getattr(model, "pendulum_fixed_observation_statistics", False) or
            getattr(model, "pendulum_curriculum", None) is not None or
            not member.env.training or not member.env.norm_obs or member.env.norm_reward):
        raise ValueError("member state does not match the declared rolling-normalization training recipe")


def make_member(member_id, config, settings, output, recipe):
    from .train_generations import TrainingMember
    member = TrainingMember(member_id, config, settings, output,
        gradient_clipping=recipe.gradient_clipping, training_boundary_m=recipe.training_boundary_m,
        boundary_failure_cost=recipe.boundary_failure_cost)
    try:
        check_member_recipe(member, recipe)
        return member
    except BaseException:
        member.close()
        raise
