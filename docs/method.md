# Temporal harmonization and fusion

At each observed source grid point, radial and tangential winds are decomposed as

`c_s = B_s + A_L,s + A_S,s`.

The background uses source-only medians in 10-km annuli. The smoothed asymmetry uses a mask-normalized horizontal Gaussian filter with 12-km standard deviation. The residual is the exact remaining perturbation on source support.

For signed offset `tau_s = t_ref - t_s`, the temporal network predicts bounded two-component tendencies and modulation parameters. Each increment contains `tau_s` and an attenuation factor `exp(-abs(tau_s)/T)`. The asymmetric increments also contain bounded signed modulation `exp(clip(tau_s*kappa))`. Zero time offset returns the original source components. The scale values are configuration parameters, not diagnosed dynamical lifetimes.

A shared set representation predicts source reliability and effective radial/tangential covariance. With source precision `P_s = q_s * inverse(Sigma_s)`, the estimate is

`c_fused = inverse(sum_s P_s) * sum_s(P_s * c_harmonized,s)`.

Only observed source cells enter these sums. The primary-reference wind is restored after combination. Cells with neither a primary-reference observation nor any source observation remain missing. Fusion does not create spatial coverage.

The fitting objective combines source and final reference reconstruction, covariance likelihoods and bounded/smooth tendency and consistency terms. Configuration contains the full coefficients. Source error estimates depend on the visible observational context and time offsets. Their inverse precision is an effective uncertainty diagnostic.

## Product fields

NetCDF output includes eastward/northward, radial/tangential and speed fields; vorticity/divergence; component uncertainties; reference mask; candidate count and support; candidate weights; raw and harmonized source winds; signed source times; and component increments.

Provenance codes are 0 for no estimate, 1 for reference observations, 2 for one source and 3 for multiple sources. Source coordinates remain identical to the target coordinates throughout the temporal model.

Primary-reference/source transitions can introduce local derivative jumps. Derivative analyses should separately flag or exclude the transition band; a 6-km exclusion is the manuscript's recommended diagnostic mask. Wind-value products retain the observed support.
