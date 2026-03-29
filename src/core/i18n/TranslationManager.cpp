// 20260323 ZJH TranslationManager — 国际化管理器实现
// 内置中英文翻译字典 + Qt QTranslator 双机制

#include "core/i18n/TranslationManager.h"


// 20260323 ZJH 获取单例实例
TranslationManager& TranslationManager::instance()
{
    static TranslationManager s_instance;  // 20260323 ZJH Meyer's Singleton
    return s_instance;
}

// 20260323 ZJH 构造函数
TranslationManager::TranslationManager()
{
    loadBuiltinTranslations();  // 20260323 ZJH 加载内置翻译
}

// 20260323 ZJH 设置语言
void TranslationManager::setLanguage(AppLanguage lang)
{
    if (m_currentLang == lang) return;  // 20260323 ZJH 无变化则跳过

    m_currentLang = lang;
    emit languageChanged(lang);  // 20260323 ZJH 通知所有监听者
}

// 20260323 ZJH 获取当前语言
AppLanguage TranslationManager::currentLanguage() const
{
    return m_currentLang;
}

// 20260323 ZJH 翻译字符串
QString TranslationManager::translate(const QString& strKey) const
{
    auto it = m_mapTranslations.find(strKey);
    if (it == m_mapTranslations.end()) {
        return strKey;  // 20260323 ZJH 找不到翻译则返回键名
    }

    if (m_currentLang == AppLanguage::Chinese) {
        return it.value().first;   // 20260323 ZJH 中文
    } else {
        return it.value().second;  // 20260323 ZJH 英文
    }
}

// 20260323 ZJH 获取语言显示名称
QString TranslationManager::languageDisplayName(AppLanguage lang)
{
    switch (lang) {
        case AppLanguage::Chinese: return QStringLiteral("简体中文");
        case AppLanguage::English: return QStringLiteral("English");
    }
    return QStringLiteral("Unknown");
}

// 20260323 ZJH 加载内置翻译字典
void TranslationManager::loadBuiltinTranslations()
{
    // 20260323 ZJH 辅助宏：添加翻译条目
    auto addTr = [this](const QString& key, const QString& zh, const QString& en) {
        m_mapTranslations.insert(key, {zh, en});
    };

    // ===== 导航栏 =====
    addTr("nav.project",    "项目",   "Project");
    addTr("nav.gallery",    "图库",   "Gallery");
    addTr("nav.image",      "图像",   "Image");
    addTr("nav.inspection", "检查",   "Inspection");
    addTr("nav.split",      "拆分",   "Split");
    addTr("nav.training",   "训练",   "Training");
    addTr("nav.evaluation", "评估",   "Evaluation");
    addTr("nav.export",     "导出",   "Export");

    // ===== 菜单 =====
    addTr("menu.file",          "文件",     "File");
    addTr("menu.edit",          "编辑",     "Edit");
    addTr("menu.view",          "视图",     "View");
    addTr("menu.help",          "帮助",     "Help");
    addTr("menu.new_project",   "新建项目", "New Project");
    addTr("menu.open_project",  "打开项目", "Open Project");
    addTr("menu.save_project",  "保存项目", "Save Project");
    addTr("menu.close_project", "关闭项目", "Close Project");
    addTr("menu.exit",          "退出",     "Exit");
    addTr("menu.undo",          "撤销",     "Undo");
    addTr("menu.redo",          "重做",     "Redo");
    addTr("menu.fullscreen",    "全屏",     "Fullscreen");
    addTr("menu.theme",         "切换主题", "Toggle Theme");
    addTr("menu.shortcuts",     "快捷键",   "Shortcuts");
    addTr("menu.about",         "关于",     "About");
    addTr("menu.settings",      "设置",     "Settings");

    // ===== 项目页 =====
    addTr("project.welcome",       "欢迎使用 OmniMatch",       "Welcome to OmniMatch");
    addTr("project.subtitle",      "纯 C++ 全流程深度学习视觉平台", "Pure C++ Deep Learning Vision Platform");
    addTr("project.new",           "新建项目",                  "New Project");
    addTr("project.open",          "打开项目",                  "Open Project");
    addTr("project.open_folder",   "打开文件夹",                "Open Folder");
    addTr("project.recent",        "最近项目",                  "Recent Projects");
    addTr("project.name",          "项目名称",                  "Project Name");
    addTr("project.task_type",     "任务类型",                  "Task Type");
    addTr("project.path",          "项目路径",                  "Project Path");
    addTr("project.created",       "创建时间",                  "Created");
    addTr("project.images",        "图像数量",                  "Images");
    addTr("project.labeled",       "已标注",                    "Labeled");

    // ===== 图库页 =====
    addTr("gallery.import_images",   "导入图像",     "Import Images");
    addTr("gallery.import_folder",   "导入文件夹",   "Import Folder");
    addTr("gallery.delete_selected", "删除选中",     "Delete Selected");
    addTr("gallery.search",          "搜索...",      "Search...");
    addTr("gallery.total",           "总计",         "Total");
    addTr("gallery.labeled",         "已标注",       "Labeled");
    addTr("gallery.unlabeled",       "未标注",       "Unlabeled");

    // ===== 训练页 =====
    addTr("train.start",           "开始训练",   "Start Training");
    addTr("train.stop",            "停止",       "Stop");
    addTr("train.pause",           "暂停",       "Pause");
    addTr("train.resume",          "继续",       "Resume");
    addTr("train.model_config",    "模型配置",   "Model Configuration");
    addTr("train.hyperparams",     "超参数",     "Hyperparameters");
    addTr("train.augmentation",    "数据增强",   "Data Augmentation");
    addTr("train.framework",       "框架",       "Framework");
    addTr("train.architecture",    "架构",       "Architecture");
    addTr("train.device",          "设备",       "Device");
    addTr("train.optimizer",       "优化器",     "Optimizer");
    addTr("train.scheduler",       "调度器",     "Scheduler");
    addTr("train.learning_rate",   "学习率",     "Learning Rate");
    addTr("train.batch_size",      "批量大小",   "Batch Size");
    addTr("train.epochs",          "训练轮次",   "Epochs");
    addTr("train.input_size",      "输入尺寸",   "Input Size");
    addTr("train.patience",        "早停耐心",   "Early Stopping Patience");

    // ===== 评估页 =====
    addTr("eval.run",              "运行评估",   "Run Evaluation");
    addTr("eval.clear",            "清除结果",   "Clear Results");
    addTr("eval.export_csv",       "导出 CSV",   "Export CSV");
    addTr("eval.export_html",      "导出报告",   "Export Report");
    addTr("eval.accuracy",         "准确率",     "Accuracy");
    addTr("eval.precision",        "精确率",     "Precision");
    addTr("eval.recall",           "召回率",     "Recall");
    addTr("eval.f1_score",         "F1 分数",    "F1 Score");
    addTr("eval.confusion_matrix", "混淆矩阵",   "Confusion Matrix");

    // ===== 导出页 =====
    addTr("export.start",          "开始导出",   "Start Export");
    addTr("export.format",         "导出格式",   "Export Format");
    addTr("export.precision",      "精度",       "Precision");
    addTr("export.output_dir",     "输出目录",   "Output Directory");

    // ===== 通用 =====
    addTr("common.ok",            "确定",   "OK");
    addTr("common.cancel",        "取消",   "Cancel");
    addTr("common.apply",         "应用",   "Apply");
    addTr("common.close",         "关闭",   "Close");
    addTr("common.browse",        "浏览",   "Browse");
    addTr("common.save",          "保存",   "Save");
    addTr("common.delete",        "删除",   "Delete");
    addTr("common.reset",         "重置",   "Reset");
    addTr("common.refresh",       "刷新",   "Refresh");
    addTr("common.all",           "全部",   "All");
    addTr("common.none",          "无",     "None");
    addTr("common.train",         "训练",   "Train");
    addTr("common.validation",    "验证",   "Validation");
    addTr("common.test",          "测试",   "Test");
    addTr("common.unassigned",    "未分配", "Unassigned");

    // ===== 任务类型 =====
    addTr("task.classification",         "图像分类",       "Classification");
    addTr("task.object_detection",       "目标检测",       "Object Detection");
    addTr("task.semantic_segmentation",  "语义分割",       "Semantic Segmentation");
    addTr("task.anomaly_detection",      "异常检测",       "Anomaly Detection");
    addTr("task.ocr",                    "OCR 文字识别",   "OCR");
    addTr("task.instance_segmentation",  "实例分割",       "Instance Segmentation");
    addTr("task.zeroshot_detection",     "零样本检测",     "Zero-Shot Detection");
    addTr("task.zeroshot_anomaly",       "零样本异常检测", "Zero-Shot Anomaly Detection");
}
