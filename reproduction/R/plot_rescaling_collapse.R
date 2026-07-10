#!/usr/bin/env Rscript
# =============================================================================
# Genuine-Folding Rescaling Collapse — Euro-Q-Exa depth sweep (EQE-day j4)
# =============================================================================
# Shows the central claim with GENUINE {1,3,5} folding only (no garbage): as
# circuit depth drives the amplified expectation values E(lambda>1) to the
# parity floor f=0, Richardson extrapolation collapses to the deterministic
# rescaling E_hat = c1*E(lambda1) = 1.875*E(lambda1), decoupling the reported
# value from the true E_ideal.
#
# Panel (a): measured E(lambda) for Trotter depths d=1,3,5; signal collapses
#            to the floor with depth.
# Panel (b): per-depth comparison of the Richardson estimate E_hat, the
#            rescaling prediction 1.875*E(lambda1), and E_ideal. When the
#            signal is destroyed (d=3,5) E_hat tracks the rescaling and is
#            decoupled from E_ideal; when retained (d=1) it does not.
#
# Input:  reproduction/data/eqe_day/j4_depth_tc{1,3,5}_raw.csv (genuine rows)
# Output: build/plots/rescaling_collapse.tex
#
# Usage:  Rscript reproduction/R/plot_rescaling_collapse.R
# =============================================================================

source("./reproduction/R/config.R")
library(patchwork)

# Richardson {1,3,5} Lagrange coefficients (floor f = 0 for parity <Z^n>).
C1 <- 15 / 8; C2 <- -5 / 4; C3 <- 3 / 8

depths <- c(tc1 = 1, tc3 = 3, tc5 = 5)
res <- tibble()
curves <- tibble()
for (tag in names(depths)) {
  f <- file.path("reproduction", "data", "eqe_day", sprintf("j4_depth_%s_raw.csv", tag))
  if (!file.exists(f)) stop("Missing: ", f)
  d <- read_csv(f, show_col_types = FALSE) %>% filter(method == "genuine")
  agg <- d %>% group_by(scale_factor) %>%
    summarise(E = mean(exp_val), ideal = first(ideal), .groups = "drop")
  e1 <- agg$E[agg$scale_factor == 1]
  e3 <- agg$E[agg$scale_factor == 3]
  e5 <- agg$E[agg$scale_factor == 5]
  ideal <- agg$ideal[1]
  ehat <- C1 * e1 + C2 * e3 + C3 * e5
  resc <- C1 * e1
  res <- bind_rows(res, tibble(
    depth = depths[[tag]],
    Richardson = ehat, Rescaling = resc, Ideal = ideal))
  curves <- bind_rows(curves, agg %>%
    mutate(depth = factor(sprintf("d=%d", depths[[tag]]),
                          levels = c("d=1", "d=3", "d=5"))))
}
cat("Per-depth summary (genuine {1,3,5}):\n"); print(res)

depth_cols <- c("d=1" = LFD$teal, "d=3" = LFD$orange, "d=5" = LFD$red)

# ── Panel (a): E(lambda) collapse to the floor ──
pa <- ggplot(curves, aes(scale_factor, E, colour = depth)) +
  geom_hline(yintercept = 0, linetype = "dotted", colour = LFD$grey,
             linewidth = 0.4) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1.3) +
  annotate("text", x = 5, y = 0.08, label = "floor $f=0$",
           hjust = 1, size = 2.4, colour = LFD$grey) +
  scale_colour_manual(values = depth_cols, name = NULL) +
  guides(colour = guide_legend(nrow = 1, keywidth = unit(2.2, "mm"),
                               keyheight = unit(2.2, "mm"),
                               override.aes = list(shape = 22, size = 2.5,
                                                   linetype = 0, stroke = 0.2,
                                                   colour = "black",
                                                   fill = unname(depth_cols)))) +
  scale_x_continuous(breaks = c(1, 3, 5), expand = expansion(mult = c(0.04, 0.1))) +
  labs(x = "noise scale $\\lambda$", y = "$E(\\lambda)$",
       title = "(a) depth destroys the signal") +
  theme_paper() + shrink_legend() +
  theme(plot.title = element_text(size = BASE.SIZE),
        legend.text = element_text(size = SMALL.SIZE - 1),
        plot.margin = margin(t = 2, r = 2, b = 0, l = 0, unit = "mm"))

# ── Panel (b): Richardson vs rescaling prediction vs ideal ──
bars <- res %>%
  pivot_longer(c(Richardson, Rescaling, Ideal),
               names_to = "quantity", values_to = "value") %>%
  mutate(depth = factor(sprintf("d=%d", depth),
                        levels = c("d=1", "d=3", "d=5")),
         quantity = factor(quantity,
                           levels = c("Richardson", "Rescaling", "Ideal")))
qcols <- c(Richardson = LFD$teal, Rescaling = LFD$orange, Ideal = LFD$red)
pb <- ggplot(bars, aes(depth, value, fill = quantity)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68,
           colour = "black", linewidth = 0.2) +
  scale_fill_manual(
    values = qcols, name = NULL,
    labels = c("Richardson", "rescaling", "ideal")) +
  guides(fill = guide_legend(nrow = 1, keywidth = unit(2.2, "mm"),
                             keyheight = unit(2.2, "mm"))) +
  labs(x = "trotter depth", y = "$\\hat E(0)$",
       title = "(b) $\\hat E(0)\\!\\to\\! 15/8\\,E(\\lambda_1)$") +
  theme_paper() + shrink_legend() +
  theme(plot.title = element_text(size = BASE.SIZE),
        legend.text = element_text(size = SMALL.SIZE - 1))

g <- pa + pb + plot_layout(widths = c(1, 1.15))
save_plot(g, "rescaling_collapse", width = COLWIDTH, height = 0.44 * COLWIDTH)
cat("Wrote build/plots/rescaling_collapse.tex\n")
