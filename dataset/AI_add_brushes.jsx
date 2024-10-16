// will use predefined file list (benchmark_list) to rasterize all vector images if this flag is true
// other wise will work in training set generate mode, it will generate images names only with number list
var BENCHMARK = false;

function randomFloat(low, high){
    return (Math.random() * (high - low) + low).toFixed(2);
}
function main(){
    // predefined file list
    var benchmark_list = [ "Art_freeform_AG_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_AG_02_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_freeform_AG_02_Liliya Larsen_norm_cleaned.svg",
                    "Art_freeform_AG_03_Liliya Larsen_norm_cleaned.svg",
                    "Art_freeform_AG_03_Maria Hegedus_norm_cleaned.svg",
                    "Art_freeform_AG_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_AG_05_Liliya Larsen_norm_cleaned.svg",
                    "Art_freeform_AG_05_Maria Hegedus_norm_cleaned.svg",
                    "Art_freeform_AP_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_AP_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_AP_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_AP_02_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_AP_02_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_AP_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_AP_03_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_AP_03_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_AP_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_AP_05_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_01_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_02_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_02_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_03_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_03_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_04_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_04_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_04_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_05_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_05_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_06_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_06_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_06_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_07_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_07_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_07_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_08_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_09_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_09_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_09_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_10_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_10_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_10_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_11_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_11_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_11_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_12_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_12_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_12_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_13_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_13_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_14_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_14_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_14_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_15_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_15_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_15_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_16_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_16_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_16_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_17_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_17_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_17_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_18_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_18_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_18_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_baseline_19_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_baseline_19_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_baseline_19_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_DR_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_DR_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_DR_01_Jonathan Velasco_norm_cleaned.svg",
                    "Art_freeform_DR_01_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_DR_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_DR_02_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_DR_02_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_DR_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_DR_03_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_DR_03_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_DR_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_DR_05_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_DR_05_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_DV_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_DV_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_DV_01_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_GL_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_GL_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_GL_01_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_GL_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_GL_02_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_GL_02_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_GW_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_GW_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_GW_01_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_JD_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_JD_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_JD_01_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_Krenz_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_Krenz_02_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_Krenz_07_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_Krenz_07_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_Krenz_07_Jonathan Velasco_norm_cleaned.svg",
                    "Art_freeform_Krenz_07_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_Krenz_10_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_Krenz_10_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_Krenz_10_Maria Hegedus_norm_cleaned.svg",
                    "Art_freeform_Krenz_26_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_Krenz_26_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_LTV_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_LTV_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_LTV_01_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_LTV_06_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_LTV_06_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_LTV_06_Liliya Larsen_norm_cleaned.svg",
                    "Art_freeform_MF_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_MF_01_Maria Hegedus_norm_cleaned.svg",
                    "Art_freeform_MF_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_MF_02_Liliya Larsen_norm_cleaned.svg",
                    "Art_freeform_MF_02_Maria Hegedus_norm_cleaned.svg",
                    "Art_freeform_MP_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_MP_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_MP_01_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_MP_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_MP_02_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_MP_02_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_PB_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_PB_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_PB_01_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_PB_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_PB_02_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_PB_06_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_PB_06_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_freeform_PB_06_Liliya Larsen_norm_cleaned.svg",
                    "Art_freeform_PB_08_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_PB_08_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_freeform_PB_09_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_freeform_PB_11_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_PB_11_Santiago Rial_norm_cleaned.svg",
                    "Art_freeform_Rui_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_freeform_Rui_01_Ge Jin_norm_cleaned.svg",
                    "Art_freeform_Rui_01_Santiago Rial_norm_cleaned.svg",
                    "Art_logo_BF_01_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_logo_BF_01_Liliya Larsen_norm_cleaned.svg",
                    "Art_logo_CA_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_logo_CA_01_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_logo_CA_01_Liliya Larsen_norm_cleaned.svg",
                    "Art_logo_JST_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_logo_JST_01_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_logo_JST_01_Liliya Larsen_norm_cleaned.svg",
                    "Art_logo_JST_05_Liliya Larsen_norm_cleaned.svg",
                    "Art_logo_VFS_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_logo_VFS_01_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_logo_VFS_01_Liliya Larsen_norm_cleaned.svg",
                    "Art_logo_VFS_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_logo_VFS_05_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_logo_VFS_05_Liliya Larsen_norm_cleaned.svg",
                    "Art_logo_VFS_08_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_logo_VFS_08_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_logo_VFS_12_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_logo_VFS_12_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_logo_VFS_12_Liliya Larsen_norm_cleaned.svg",
                    "Art_logo_VFS_15_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_logo_VFS_15_Liliya Larsen_norm_cleaned.svg",
                    "Art_logo_VFS_16_Branislav Mirkovic_norm_cleaned.svg",
                    "Art_logo_VFS_16_Diego Barrionuevo_norm_cleaned.svg",
                    "Art_logo_VFS_16_Liliya Larsen_norm_cleaned.svg",
                    "Ind_architecture_AST_01_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_AST_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_AST_02_Maria Hegedus_norm_cleaned.svg",
                    "Ind_architecture_AST_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_AST_05_Diego Barrionuevo_norm_cleaned.svg",
                    "Ind_architecture_baseline_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_baseline_01_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_baseline_01_Santiago Rial_norm_cleaned.svg",
                    "Ind_architecture_baseline_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_baseline_02_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_baseline_02_Santiago Rial_norm_cleaned.svg",
                    "Ind_architecture_baseline_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_baseline_03_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_baseline_03_Santiago Rial_norm_cleaned.svg",
                    "Ind_architecture_baseline_04_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_baseline_04_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_baseline_04_Santiago Rial_norm_cleaned.svg",
                    "Ind_architecture_baseline_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_baseline_05_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_baseline_05_Santiago Rial_norm_cleaned.svg",
                    "Ind_architecture_baseline_06_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_baseline_06_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_baseline_06_Santiago Rial_norm_cleaned.svg",
                    "Ind_architecture_GD_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_GD_03_Liliya Larsen_norm_cleaned.svg",
                    "Ind_architecture_GD_05_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_GD_05_Maria Hegedus_norm_cleaned.svg",
                    "Ind_architecture_JJ_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_JJ_02_Liliya Larsen_norm_cleaned.svg",
                    "Ind_architecture_JJ_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_JJ_03_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_JJ_03_Liliya Larsen_norm_cleaned.svg",
                    "Ind_architecture_JJ_04_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_architecture_JJ_04_Liliya Larsen_norm_cleaned.svg",
                    "Ind_architecture_NMC_01_Liliya Larsen_norm_cleaned.svg",
                    "Ind_architecture_SG_01_Santiago Rial_norm_cleaned.svg",
                    "Ind_architecture_TU_01_Liliya Larsen_norm_cleaned.svg",
                    "Ind_architecture_TU_02_Ge Jin_norm_cleaned.svg",
                    "Ind_architecture_TU_02_Santiago Rial_norm_cleaned.svg",
                    "Ind_fashion_HF_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_HF_01_Ge Jin_norm_cleaned.svg",
                    "Ind_fashion_HF_01_Santiago Rial_norm_cleaned.svg",
                    "Ind_fashion_HF_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_HF_02_Liliya Larsen_norm_cleaned.svg",
                    "Ind_fashion_HF_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_HF_05_Ge Jin_norm_cleaned.svg",
                    "Ind_fashion_LB_01_Diego Barrionuevo_norm_cleaned.svg",
                    "Ind_fashion_LB_01_Liliya Larsen_norm_cleaned.svg",
                    "Ind_fashion_ML_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_ML_10_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_ML_10_Diego Barrionuevo_norm_cleaned.svg",
                    "Ind_fashion_ML_10_Liliya Larsen_norm_cleaned.svg",
                    "Ind_fashion_ML_11_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_ML_11_Liliya Larsen_norm_cleaned.svg",
                    "Ind_fashion_RB_04_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_RB_04_Ge Jin_norm_cleaned.svg",
                    "Ind_fashion_RB_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_RB_05_Ge Jin_norm_cleaned.svg",
                    "Ind_fashion_RB_05_Liliya Larsen_norm_cleaned.svg",
                    "Ind_fashion_RB_10_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_RB_10_Ge Jin_norm_cleaned.svg",
                    "Ind_fashion_RB_10_Liliya Larsen_norm_cleaned.svg",
                    "Ind_fashion_RB_18_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_fashion_RB_18_Ge Jin_norm_cleaned.svg",
                    "Ind_fashion_RB_18_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_AS_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_AS_03_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_AS_03_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_AS_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_AS_05_Ge Jin_norm_cleaned.svg",
                    "Ind_product_AS_05_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_AS_11_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_AS_11_Ge Jin_norm_cleaned.svg",
                    "Ind_product_AS_11_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_AS_12_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_AS_14_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_AS_14_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_AS_14_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_AS_15_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_AS_15_Ge Jin_norm_cleaned.svg",
                    "Ind_product_AS_15_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_AS_18_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_AS_18_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_AS_18_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_AS_20_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_AS_20_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_AS_20_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_baseline_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_01_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_01_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_02_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_02_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_03_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_03_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_04_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_04_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_04_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_05_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_05_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_06_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_06_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_06_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_07_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_07_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_07_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_08_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_08_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_08_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_09_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_09_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_09_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_10_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_10_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_10_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_11_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_11_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_11_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_12_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_12_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_12_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_13_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_13_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_13_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_14_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_14_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_14_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_baseline_15_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_baseline_15_Ge Jin_norm_cleaned.svg",
                    "Ind_product_baseline_15_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_GW_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_GW_01_Ge Jin_norm_cleaned.svg",
                    "Ind_product_GW_01_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_GW_02_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_GW_02_Ge Jin_norm_cleaned.svg",
                    "Ind_product_GW_02_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_GW_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_GW_03_Ge Jin_norm_cleaned.svg",
                    "Ind_product_GW_03_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_GW_04_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_GW_04_Ge Jin_norm_cleaned.svg",
                    "Ind_product_GW_04_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_GW_06_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_GW_06_Ge Jin_norm_cleaned.svg",
                    "Ind_product_GW_06_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_GW_09_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_GW_09_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_GW_09_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_JM_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_JM_03_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_JM_03_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_JM_05_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_JM_05_Ge Jin_norm_cleaned.svg",
                    "Ind_product_JM_05_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_JM_07_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_JM_07_Ge Jin_norm_cleaned.svg",
                    "Ind_product_JM_07_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_JM_08_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_JM_08_Ge Jin_norm_cleaned.svg",
                    "Ind_product_JM_08_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_JM_09_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_JM_09_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_MMX_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_MMX_01_Ge Jin_norm_cleaned.svg",
                    "Ind_product_MMX_01_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_MMX_02_Ge Jin_norm_cleaned.svg",
                    "Ind_product_MMX_02_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_PM_01_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_PM_03_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_PM_03_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_PM_03_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_PM_04_Ge Jin_norm_cleaned.svg",
                    "Ind_product_PM_04_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_PM_07_Ge Jin_norm_cleaned.svg",
                    "Ind_product_PM_07_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_PM_12_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_PM_12_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_PM_12_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_PM_14_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_PM_14_Ge Jin_norm_cleaned.svg",
                    "Ind_product_PM_14_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_PM_25_Ge Jin_norm_cleaned.svg",
                    "Ind_product_PM_25_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_PM_43_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_PM_43_Ge Jin_norm_cleaned.svg",
                    "Ind_product_PM_43_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_PM_46_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_PM_46_Ge Jin_norm_cleaned.svg",
                    "Ind_product_PM_46_Liliya Larsen_norm_cleaned.svg",
                    "Ind_product_PM_47_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_PM_47_Ge Jin_norm_cleaned.svg",
                    "Ind_product_PM_47_Maria Hegedus_norm_cleaned.svg",
                    "Ind_product_SP_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_SP_01_Ge Jin_norm_cleaned.svg",
                    "Ind_product_SP_01_Santiago Rial_norm_cleaned.svg",
                    "Ind_product_TI_01_Branislav Mirkovic_norm_cleaned.svg",
                    "Ind_product_TI_01_Ge Jin_norm_cleaned.svg",
                    "Ind_product_TI_01_Santiago Rial_norm_cleaned.svg"];
    // add copied brushes into the target svg files and save them into AI files
    var pathStrSplit = $.fileName.split('/');
    pathStrSplit.pop(); // pop out the current file name, we don't need it
    pathStrSplit.pop();

    var currPath = pathStrSplit.join('/');
    var brushTemplate = currPath + "/dataset/brush02.ai";
    if (BENCHMARK){
        var svgPath = currPath + "/data/benchmark/256_long/svg/";
        var pngPath = currPath + "/data/benchmark/256_long/png_hard/";
    }
    else{
        var svgPath = currPath + "/data/full/svg/";
        var pngPath = currPath + "/data/full/png/";
    }
    
    
    // set up the export option
    var options = new ImageCaptureOptions();
    options.artBoardClipping = true;
    options.resolution = 72; // dpi 150
    options.antiAliasing = true;
    options.matte = false;
    options.horizontalScale = 100;
    options.verticalScale = 100;
    options.transparency = true; // I hope this could be helpful when adding paper textures
                            

    var opts = new ExportOptionsPNG24();
    opts.antiAliasing = true;
    opts.transparency = true;
    opts.artBoardClipping = true;
    opts.horizontalScale = 100;
    opts.verticalScale = 100;
    opts.resolution = 150;
    // var type = ExportType.PNG24;
    
    var brushNames = [null, "Calligraphic Brush 1", "BrushPen 111", "HEJ_BLACK_STROKE_01", "HEJ_TRUE_GRIS_M_STROKE_01", 
        "Lino Cut 8", "BrushPen 42", "Charcoal_smudged_3", "HEJ_ANGRY_STROKE_03", 'HEJ_TRUE_GRIS_M_STROKE_04', 'Graphite - B_short',
        "Comic Book_Contrast 3"];
    // todo: 104, 002, 001
    // [basic, Calligraphic Brush 1, BrushPen 42, Charcoal_smudged_3, HEJ_TRUE_GRIS_M_STROKE_04, BrushPen 111, Comic Book_Contrast 3]
    var brushes_train = [0, 1, 6, 7, 9, 2, 11];
    var brushes_width_train = [[1, 5], [0.5, 1], [0.05, 0.2], [0.01, 0.15], [0.5, 5], [0.1, 1], [0.1, 1]];

    // var brushes_eval = [1, 7, 2];
    // var brushes_width_eval = [0.5, 0.08, 0.5];
    
    var brushes_eval = [6, 7, 9, 2, 11];
    var brushes_width_eval = [[0.05, 0.1], [0.01, 0.07], [0.5, 1], [0.1, 0.5], [0.1, 0.5]];
    
    var brush = new File(brushTemplate);
    var foundBrush = false;
    /////////////////////////////
    // 0: basic, 
    // 1: Calligraphic Brush 1, 
    // 2: BrushPen 42, 
    // 3: Charcoal_smudged_3, 
    // 4: HEJ_TRUE_GRIS_M_STROKE_04, 
    // 5: BrushPen 111, 
    // 6: Comic Book_Contrast 3
    // change this variable to select brush type above before you run this script everytime
    var random = 5;
    /////////////////////////////
    var brushIdx = null;
    if (BENCHMARK)
        var fileNum = 5;
    else
        var fileNum = 20;
    var fileCounter = 0;
    var processedCounter = 0;
    // open fileNums svg files at once
    // uncomment this block if we are generating benchmark set
    // for (var i = 0; i < benchmark_list.length; i = i + fileNum){
    //     var svgList = [];
    //     var svgDocList = [];
    //     for (var k = 0; k < fileNum; k++){
    //         var svgInput = svgPath + benchmark_list[i + k];
    //         var svg = new File(svgInput);
    //         if (svg.exists){
    //             svgList.push(svg);
    //         }
    //     }

    // uncomment this block if we are generating training set
    for (var i = 0; i < 9999999; i = i + fileNum){
        // create the file list
        var svgList = [];
        var svgDocList = [];
        for (var j = 0; j < 9999; j++){
            // skip 1/4
            // if ((i+j) % 10 == 0 | (i+j) % 10 == 1 | (i+j) % 10 == 2) continue;
            // skip 3/4
            if ((i+j) % 10 == 3 | (i+j) % 10 == 4 | (i+j) % 10 == 5 | (i+j) % 10 == 6 |  (i+j) % 10 == 7 | (i+j) % 10 == 8 | (i+j) % 10 == 9) continue;
            if (fileCounter >= fileNum) break;
                    var svgInput = svgPath + ("0000000" + (i + j)).slice(-7) + ".svg";
            var svg = new File(svgInput);
            if (svg.exists){
                svgList.push(svg);
                fileCounter++;
            }
        }
        fileCounter = 0;
        
        // apply brushes to svg files and export
        for (var j = 0; j < svgList.length; j++){
            // prepare file objects
            if (BENCHMARK){
                random = Math.floor(Math.random() * brushes_eval.length);
                brushIdx = brushes_eval[random];
            }
            else{
                // change value here manually to select which brush we want to use
                brushIdx = brushes_train[random]; // idx of bursh types, 0 means basic line    
            }
            var svg = svgList[j];
            var name = svg.name.split('.')[0];
            if (BENCHMARK){
                var pngInput = pngPath + name + ".png";
            }
            else{
                var pngInput = pngPath + ("00" + brushIdx).slice(-2) + "/" + name + ".png";    
            }
            
            var png = new File(pngInput);
            if (png.exists) continue;
            if (svg.exists){
                if (!brush.exists & brushIdx != 0){
                    alert("Can't find the brush template, please check if the file exists.", "Missing brush template");
                    break;
                }
                // check the opened documents, see if the brush template is opened
                for (var k = 0; k < app.documents.length; k++){
                    if (app.documents[k].name == "brush02.ai") foundBrush = true;
                }
                // if not, open the brush template and copy the brush info
                if (!foundBrush){
                    var docBrushes = app.open(brush);
                    docBrushes.activate();
                    docBrushes.selectObjectsOnActiveArtboard();
                    app.executeMenuCommand("copy");
                    foundBrush = true;
                }
                var docSVG = app.open(svg);
                svgDocList.push(docSVG);
                if (app.activeDocument.name != docSVG.name) docSVG.activate();
                // thanks for https://community.adobe.com/t5/illustrator-discussions/script-for-copy-and-paste-object-from-one-ai-file-to-another-ai-file/td-p/8059026
                if (brushIdx != 0){
                    app.executeMenuCommand("paste");
                    // seems sleep not helps
                    // sleep(2000);
                    app.executeMenuCommand("clear");
                }
                // build up the mapping from the brush name to brush index
                var brushNametoIdx = {0: null};
                for (var l =0; l < brushNames.length; l ++){
                    for (var m = 0; m < app.activeDocument.brushes.length; m++){
                        if (app.activeDocument.brushes[m].name == brushNames[l])
                            brushNametoIdx[brushNames[l]] = m;
                    }
                }
                // apply brush
                docSVG.selectObjectsOnActiveArtboard();
                var width = null;
                var low = null;
                var high = null;
                if (BENCHMARK){
                    low = brushes_width_eval[random][0];
                    high = brushes_width_eval[random][1];
                    width = randomFloat(low, high);
                }
                else{
                    low = brushes_width_train[random][0];
                    high = brushes_width_train[random][1];
                    width = randomFloat(low, high);
                }

                toPngWithBrush(brushNametoIdx[brushNames[brushIdx]], docSVG, png, options, width, low, high);
                processedCounter = processedCounter + 1;
            }
        }
        
        // close all files
        for (var j = 0; j < svgDocList.length; j++){
            var docSVG = svgDocList[j];
            // app.executeCommand(10200);
            docSVG.close(SaveOptions.DONOTSAVECHANGES);
        }
        // stop every 4k images, to release RAM
        if (processedCounter > 50000) break;
    }
}


function sleep(milliseconds) {
    const date = Date.now();
    var currentDate = null;
    do {
      currentDate = Date.now();
    } while (currentDate - date < milliseconds);
  }

function toPngWithBrush(Idx, docSVG, png, options, width, low, high){
    // select the artboard need to export
    var activeAB = docSVG.artboards[docSVG.artboards.getActiveArtboardIndex()];
    // select a brush
    if (Idx != null){
        var brushType = app.activeDocument.brushes[Idx];
        // apply the brush
        for (var j = 0; j < app.activeDocument.pageItems.length; j++){
            brushType.applyTo(app.activeDocument.pageItems[j]);
            if (BENCHMARK == false){
                width = randomFloat(low, high);
            }
            app.activeDocument.pageItems[j].strokeWidth = width;
            
        }
    }
    else{
        // increase the stroke width when exporting with basic stroke style
        for (var j = 0; j < app.activeDocument.pageItems.length; j++){
            if (BENCHMARK == false){
                width = randomFloat(low, high);
            }
            app.activeDocument.pageItems[j].strokeWidth = width;
        }
    }
    // export to png
    try {
        docSVG.imageCapture(png, activeAB.artboardRect, options);
    } catch (e) {}
    // export can't specify the image resolution
    // try {
    //     docSVG.exportFile(png, type, opts);
    //     docSVG.close(SaveOptions.DONOTSAVECHANGES);
    // } catch (e) {}
}

// run the script
try{
    main();
} catch (e) {}