#!/usr/bin/env Rscript
# =============================================================================
# Richardson Extrapolation Failure Modes — E(λ) Curves
# =============================================================================
# Reads horoscope_circuits.csv (from horoscope_mechanism.py Part E) and
# data/qexa_hardware.csv (static real-hardware data) to produce the main
# Layer 3 figure: Lagrange polynomial E(λ) curves showing three failure
# regimes (retained / destroyed / inverted).
#
# Input:  results/horoscope_circuits.csv, data/qexa_hardware.csv
# Output: plots/horoscope_sweep.pdf
#
# Usage:
#   cd reproduction && Rscript R/plot_extrapolation.R
# =============================================================================

source("./reproduction/R/config.R")

# =============================================================================
# Load data
# =============================================================================
circuits_path <- file.path("build", "results", "horoscope_circuits.csv")
qexa_path     <- file.path("reproduction", "data", "qexa_hardware.csv")

if (!file.exists(circuits_path)) {
  stop("Missing: ", circuits_path,
       "\nRun: python scripts/horoscope_mechanism.py first")
}

df_sim <- read_csv(circuits_path, show_col_types = FALSE)
cat("Loaded simulation data:\n")
print(df_sim %>% select(circuit, n_cx, E1, E3, E5, E_mit_gen, rho_gen))

# Reshape: one row per curve (genuine + garbage for Grover, genuine for others)
curves <- tibble()

# QFT Mirror — genuine only
qft <- df_sim %>% filter(circuit == "QFT Mirror 6q")
curves <- bind_rows(curves, tibble(
  name = "QFT Mirror 6q", E1 = qft$E1, E3 = qft$E3, E5 = qft$E5,
  ideal = qft$E_ideal, colour = LFD$teal, lty = "solid", shape = 17
))

# Trotter — genuine only
trot <- df_sim %>% filter(circuit == "Trotter 4q")
curves <- bind_rows(curves, tibble(
  name = "Trotter 4q", E1 = trot$E1, E3 = trot$E3, E5 = trot$E5,
  ideal = trot$E_ideal, colour = LFD$purple, lty = "solid", shape = 15
))

# Grover — genuine
grov <- df_sim %>% filter(circuit == "Grover 6q")
curves <- bind_rows(curves, tibble(
  name = "Grover 6q (genuine)", E1 = grov$E1, E3 = grov$E3, E5 = grov$E5,
  ideal = grov$E_ideal, colour = LFD$orange, lty = "solid", shape = 16
))

# Grover — garbage
curves <- bind_rows(curves, tibble(
  name = "Grover 6q (garbage)", E1 = grov$E1_garb, E3 = grov$E3_garb, E5 = grov$E5_garb,
  ideal = grov$E_ideal, colour = LFD$grey, lty = "dotted", shape = 1
))

# QExa hardware (static data if available)
if (file.exists(qexa_path)) {
  df_qexa <- read_csv(qexa_path, show_col_types = FALSE)
  qexa <- df_qexa[1, ]
  curves <- bind_rows(curves, tibble(
    name = "QExa hardware", E1 = qexa$E1, E3 = qexa$E3, E5 = qexa$E5,
    ideal = qexa$E_ideal, colour = LFD$red, lty = "solid", shape = 18
  ))
}

desired_curve_order <- c(
  "QFT Mirror 6q",
  "Grover 6q (genuine)",
  "Grover 6q (garbage)",
  "Trotter 4q",
  "QExa hardware"
)
desired_curve_colours <- c(
  "QFT Mirror 6q" = LFD$black,
  "Grover 6q (genuine)" = LFD$orange,
  "Grover 6q (garbage)" = LFD$grey,
  "Trotter 4q" = LFD$teal,
  "QExa hardware" = LFD$red
)

curves <- curves %>%
  mutate(
    order_idx = match(name, desired_curve_order),
    colour = unname(desired_curve_colours[name])
  ) %>%
  arrange(order_idx) %>%
  select(-order_idx)

# =============================================================================
# Richardson coefficients for λ = {1, 3, 5}
# =============================================================================
lambdas <- c(1, 3, 5)

lagrange_coeff <- function(lambdas, target = 0) {
  K <- length(lambdas)
  c_vec <- numeric(K)
  for (i in 1:K) {
    li <- 1.0
    for (j in 1:K) {
      if (j != i) li <- li * (target - lambdas[j]) / (lambdas[i] - lambdas[j])
    }
    c_vec[i] <- li
  }
  c_vec
}
C_rich <- lagrange_coeff(lambdas)

# Compute mitigated values and ρ
curves <- curves %>%
  mutate(
    E_mit = C_rich[1] * E1 + C_rich[2] * E3 + C_rich[3] * E5,
    rho   = (E_mit - E1) / (ideal - E1)
  )

cat("\nCurve summary:\n")
curves %>% select(name, E1, E3, E5, E_mit, rho) %>% print()

# =============================================================================
# Lagrange polynomial for smooth plotting
# =============================================================================
poly_eval <- function(E_vec, lam_vec, lam_target) {
  K <- length(lam_vec)
  n <- length(lam_target)
  result <- rep(0, n)
  for (i in 1:K) {
    li <- rep(1, n)
    for (j in 1:K) {
      if (j != i) li <- li * (lam_target - lam_vec[j]) / (lam_vec[i] - lam_vec[j])
    }
    result <- result + E_vec[i] * li
  }
  result
}

lam_fine <- seq(-0.3, 5.5, length.out = 300)

df_interp <- tibble()
df_extrap <- tibble()
df_points <- tibble()
df_stars  <- tibble()
df_ideals <- tibble()

for (i in 1:nrow(curves)) {
  row <- curves[i, ]
  E_vec <- c(row$E1, row$E3, row$E5)
  poly_vals <- poly_eval(E_vec, lambdas, lam_fine)

  mask_i <- lam_fine >= 1 & lam_fine <= 5
  df_interp <- bind_rows(df_interp, tibble(
    lambda = lam_fine[mask_i], E = poly_vals[mask_i],
    name = row$name, colour = row$colour, lty = row$lty
  ))

  mask_e <- lam_fine < 1
  df_extrap <- bind_rows(df_extrap, tibble(
    lambda = lam_fine[mask_e], E = poly_vals[mask_e],
    name = row$name, colour = row$colour
  ))

  df_points <- bind_rows(df_points, tibble(
    lambda = lambdas, E = E_vec,
    name = row$name, colour = row$colour, shape = row$shape
  ))

  df_stars <- bind_rows(df_stars, tibble(
    lambda = 0, E = row$E_mit,
    name = row$name, colour = row$colour
  ))

  if (!grepl("garbage", row$name, ignore.case = TRUE)) {
    df_ideals <- bind_rows(df_ideals, tibble(
      lambda = -0.25, E = row$ideal,
      name = row$name, colour = row$colour
    ))
  }
}

# Convert to ordered factors
curve_order <- curves$name
df_interp$name <- factor(df_interp$name, levels = curve_order)
df_extrap$name <- factor(df_extrap$name, levels = curve_order)
df_points$name <- factor(df_points$name, levels = curve_order)
df_stars$name  <- factor(df_stars$name,  levels = curve_order)
if (nrow(df_ideals) > 0) {
  df_ideals$name <- factor(df_ideals$name, levels = curve_order)
}

col_map   <- setNames(curves$colour, curves$name)
shape_map <- setNames(curves$shape, curves$name)
lty_map   <- setNames(curves$lty, curves$name)
legend_labels <- c(
  "QFT Mirror 6q" = "QFT Mirror 6q",
  "Grover 6q (genuine)" = "Grover 6q\n(genuine)",
  "Grover 6q (garbage)" = "Grover 6q\n(garbage)",
  "Trotter 4q" = "Trotter 4q",
  "QExa hardware" = "QExa hardware"
)

# =============================================================================
# Build Plot
# =============================================================================
noise_floor <- 1 / 64  # 6-qubit: 1/2^6

p <- ggplot() +
  geom_hline(yintercept = noise_floor, colour = LFD$grey, linetype = "dotted",
             linewidth = 0.4) +
  geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey85", linewidth = 0.3) +

  geom_line(data = df_extrap, aes(x = lambda, y = E, colour = name),
            linetype = "dashed", linewidth = 0.6, alpha = 0.6,
            show.legend = FALSE) +

  geom_line(data = df_interp, aes(x = lambda, y = E, colour = name,
                                   linetype = name),
            linewidth = 0.8) +

  geom_point(data = df_points, aes(x = lambda, y = E, colour = name,
                                    shape = name),
             size = 2, stroke = 0.5) +

  geom_point(data = df_stars, aes(x = lambda, y = E, colour = name),
             shape = 8, size = 2.5, stroke = 0.6, show.legend = FALSE) +

  geom_point(data = df_ideals, aes(x = lambda, y = E, colour = name),
             shape = 4, size = 2.5, stroke = 0.8, show.legend = FALSE) +

  annotate("text", x = 5.35, y = noise_floor, label = "$1/2^n$",
           size = 2.5, colour = LFD$grey, vjust = -0.5) +

  scale_colour_manual(values = col_map, breaks = curve_order,
                      labels = legend_labels[curve_order], name = NULL) +
  scale_linetype_manual(values = lty_map, breaks = curve_order,
                        labels = legend_labels[curve_order], name = NULL) +
  scale_shape_manual(values = shape_map, breaks = curve_order,
                     labels = legend_labels[curve_order], name = NULL) +

  scale_x_continuous(
    name = "Scale factor $\\lambda$",
    breaks = 0:5,
    labels = c("0\n(extrap.)", "1", "2", "3", "4", "5"),
    limits = c(-0.5, 5.7),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    name = "$E(\\lambda)$",
    limits = c(-0.18, 1.10),
    breaks = seq(-0.2, 1.0, 0.2),
    expand = c(0, 0)
  ) +

  guides(
    colour   = guide_legend(order = 1, nrow = 2, byrow = TRUE,
                            override.aes = list(size = 1.8, linewidth = 0.7)),
    linetype = guide_legend(order = 1, nrow = 2, byrow = TRUE),
    shape    = guide_legend(order = 1, nrow = 2, byrow = TRUE)
  ) +

  theme_paper() +
  theme(
    legend.position = "top",
    legend.justification = c(0, 1),
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.background = element_blank(),
    legend.key.width = unit(0.8, "cm"),
    legend.key.height = unit(0.32, "cm"),
    legend.margin = margin(0, 0, 0, 0),
    legend.spacing.x = unit(2, "pt"),
    legend.spacing.y = unit(1, "pt"),
    legend.text = element_text(size = SMALL.SIZE)
  )

# =============================================================================
# Save
# =============================================================================
save_plot(p, "horoscope_sweep", width = COLWIDTH, height = 0.70 * COLWIDTH)
