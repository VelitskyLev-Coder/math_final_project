TEX_DIR := tex
TEX_MAIN := linked_beverton_holt_fishing.tex
PDFLATEX := pdflatex
PYTHON := ./.venv/Scripts/python.exe
FISHING_PLOT := $(TEX_DIR)/plots/fishing_equilibrium_r1_4_k1.pdf
YIELD_PLOT := $(TEX_DIR)/plots/fishing_yield_r1_4_k1.pdf
LINKED_NO_FISHING_PLOT := $(TEX_DIR)/plots/linked_no_fishing_extinction_m0_2.pdf
LINKED_CONNECTION_PLOT := $(TEX_DIR)/plots/linked_connection_effect_ra1_8_rb1_2.pdf
LINKED_FISHING_YIELD_PLOT := $(TEX_DIR)/plots/linked_fishing_yield_ra1_8_rb1_2_m0_2.pdf
LINKED_COMPLETE_HARVEST_YIELD_PLOT := $(TEX_DIR)/plots/linked_complete_harvest_yield_examples.pdf
LINKED_COMPLETE_HARVEST_CORNER_PLOT := $(TEX_DIR)/plots/linked_complete_harvest_corner_example.pdf
LINKED_NO_TAKE_YIELD_PLOT := $(TEX_DIR)/plots/linked_no_take_yield_examples.pdf
LINKED_EXTINCTION_PLOTS := \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb0_8_m0_3.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb1_2_m0_1.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb1_2_m0_3.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb1_2_m0_0.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb1_2_m1_0.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_8_rb1_2_m0_1.pdf
PLOTS := $(FISHING_PLOT) $(YIELD_PLOT) $(LINKED_NO_FISHING_PLOT) $(LINKED_CONNECTION_PLOT) $(LINKED_FISHING_YIELD_PLOT) $(LINKED_COMPLETE_HARVEST_YIELD_PLOT) $(LINKED_COMPLETE_HARVEST_CORNER_PLOT) $(LINKED_NO_TAKE_YIELD_PLOT) $(LINKED_EXTINCTION_PLOTS)
PLOT_COMMON := plot_generators/common.py

.PHONY: all pdf clean

all: pdf

pdf: $(PLOTS)
	cd $(TEX_DIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(TEX_MAIN)
	cd $(TEX_DIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(TEX_MAIN)

$(FISHING_PLOT): plot_generators/fishing_equilibrium.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.fishing_equilibrium

$(YIELD_PLOT): plot_generators/fishing_yield.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.fishing_yield

$(LINKED_NO_FISHING_PLOT): plot_generators/linked_no_fishing_extinction.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_no_fishing_extinction

$(LINKED_CONNECTION_PLOT): plot_generators/linked_connection_effect.py linked_model.py beverton_holt.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_connection_effect

$(LINKED_FISHING_YIELD_PLOT): plot_generators/linked_fishing_yield.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_fishing_yield

$(LINKED_COMPLETE_HARVEST_YIELD_PLOT): plot_generators/linked_complete_harvest_yield_examples.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_complete_harvest_yield_examples

$(LINKED_COMPLETE_HARVEST_CORNER_PLOT): plot_generators/linked_complete_harvest_corner_example.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_complete_harvest_corner_example

$(LINKED_NO_TAKE_YIELD_PLOT): plot_generators/linked_no_take_yield_examples.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_no_take_yield_examples

$(LINKED_EXTINCTION_PLOTS): plot_generators/linked_extinction_maps.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_extinction_maps

clean:
	cd $(TEX_DIR) && powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue *.aux,*.log,*.out,*.toc,*.fls,*.fdb_latexmk,*.synctex.gz,*.pdf"
	powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue '$(TEX_DIR)/plots/*.pdf'"
