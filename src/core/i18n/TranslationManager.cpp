// 20260323 ZJH TranslationManager — 国际化管理器实现
// 内置中/英/日/德四语翻译字典 + Qt QTranslator 双机制
// 20260330 ZJH 扩展: 日语(ja_JP)/德语(de_DE) + 系统 locale 自动检测 + 动态切换

#include "core/i18n/TranslationManager.h"

#include <QLocale>   // 20260330 ZJH 系统 locale 检测


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

    // 20260330 ZJH 自动检测系统语言并设置为默认语言
    m_currentLang = detectSystemLanguage();
}

// 20260323 ZJH 设置语言（运行时动态切换，无需重启）
void TranslationManager::setLanguage(AppLanguage lang)
{
    if (m_currentLang == lang) return;  // 20260323 ZJH 无变化则跳过

    m_currentLang = lang;  // 20260330 ZJH 切换当前语言
    emit languageChanged(lang);  // 20260323 ZJH 通知所有监听者刷新 UI 文本
}

// 20260323 ZJH 获取当前语言
AppLanguage TranslationManager::currentLanguage() const
{
    return m_currentLang;
}

// 20260330 ZJH 翻译字符串（根据当前语言从 4 语言数组中取值）
QString TranslationManager::translate(const QString& strKey) const
{
    auto it = m_mapTranslations.find(strKey);
    if (it == m_mapTranslations.end()) {
        return strKey;  // 20260323 ZJH 找不到翻译则返回键名
    }

    // 20260330 ZJH 用语言枚举的整数值索引翻译数组
    int nIdx = static_cast<int>(m_currentLang);
    if (nIdx < 0 || nIdx > 3) {
        nIdx = 1;  // 20260330 ZJH 越界保护，回退到英文
    }
    return it.value().arrTexts[nIdx];
}

// 20260330 ZJH 获取语言显示名称
QString TranslationManager::languageDisplayName(AppLanguage lang)
{
    switch (lang) {
        case AppLanguage::Chinese:  return QStringLiteral("简体中文");
        case AppLanguage::English:  return QStringLiteral("English");
        case AppLanguage::Japanese: return QStringLiteral("日本語");
        case AppLanguage::German:   return QStringLiteral("Deutsch");
    }
    return QStringLiteral("Unknown");
}

// 20260330 ZJH 获取所有支持的语言列表
QVector<AppLanguage> TranslationManager::supportedLanguages()
{
    return {
        AppLanguage::Chinese,
        AppLanguage::English,
        AppLanguage::Japanese,
        AppLanguage::German
    };
}

// 20260330 ZJH 从系统 locale 自动检测语言
AppLanguage TranslationManager::detectSystemLanguage()
{
    // 20260330 ZJH 获取系统 locale 名称，如 "zh_CN", "en_US", "ja_JP", "de_DE"
    QString strLocale = QLocale::system().name();

    // 20260330 ZJH 按语言代码前缀匹配
    if (strLocale.startsWith(QStringLiteral("zh"))) {
        return AppLanguage::Chinese;   // 20260330 ZJH zh_CN / zh_TW → 中文
    } else if (strLocale.startsWith(QStringLiteral("ja"))) {
        return AppLanguage::Japanese;  // 20260330 ZJH ja_JP → 日文
    } else if (strLocale.startsWith(QStringLiteral("de"))) {
        return AppLanguage::German;    // 20260330 ZJH de_DE / de_AT → 德文
    } else {
        return AppLanguage::English;   // 20260330 ZJH 其他所有 locale → 英文
    }
}

// 20260330 ZJH 语言枚举 → locale 字符串
QString TranslationManager::languageToLocale(AppLanguage lang)
{
    switch (lang) {
        case AppLanguage::Chinese:  return QStringLiteral("zh_CN");
        case AppLanguage::English:  return QStringLiteral("en_US");
        case AppLanguage::Japanese: return QStringLiteral("ja_JP");
        case AppLanguage::German:   return QStringLiteral("de_DE");
    }
    return QStringLiteral("en_US");  // 20260330 ZJH 默认回退英文
}

// 20260330 ZJH locale 字符串 → 语言枚举
AppLanguage TranslationManager::localeToLanguage(const QString& strLocale)
{
    if (strLocale.startsWith(QStringLiteral("zh"))) return AppLanguage::Chinese;
    if (strLocale.startsWith(QStringLiteral("ja"))) return AppLanguage::Japanese;
    if (strLocale.startsWith(QStringLiteral("de"))) return AppLanguage::German;
    return AppLanguage::English;  // 20260330 ZJH 默认回退英文
}

// 20260330 ZJH 加载内置翻译字典（4 种语言）
void TranslationManager::loadBuiltinTranslations()
{
    // 20260330 ZJH 辅助 lambda：添加 4 语言翻译条目
    // 参数: key - 翻译键名, zh - 中文, en - 英文, ja - 日文, de - 德文
    auto addTr = [this](const QString& key, const QString& zh, const QString& en,
                        const QString& ja, const QString& de) {
        TranslationEntry entry;
        entry.arrTexts[0] = zh;  // 20260330 ZJH 索引 0 = Chinese
        entry.arrTexts[1] = en;  // 20260330 ZJH 索引 1 = English
        entry.arrTexts[2] = ja;  // 20260330 ZJH 索引 2 = Japanese
        entry.arrTexts[3] = de;  // 20260330 ZJH 索引 3 = German
        m_mapTranslations.insert(key, entry);
    };

    // ===== 导航栏（8 个页面标签）=====
    addTr("nav.project",    "项目",   "Project",    "プロジェクト",       "Projekt");
    addTr("nav.gallery",    "图库",   "Gallery",    "ギャラリー",         "Galerie");
    addTr("nav.image",      "图像",   "Image",      "画像",               "Bild");
    addTr("nav.inspection", "检查",   "Inspection", "検査",               "Inspektion");
    addTr("nav.split",      "拆分",   "Split",      "分割",               "Aufteilung");
    addTr("nav.training",   "训练",   "Training",   "トレーニング",       "Training");
    addTr("nav.evaluation", "评估",   "Evaluation", "評価",               "Bewertung");
    addTr("nav.export",     "导出",   "Export",     "エクスポート",       "Export");

    // ===== 菜单 =====
    addTr("menu.file",          "文件",     "File",           "ファイル",         "Datei");
    addTr("menu.edit",          "编辑",     "Edit",           "編集",             "Bearbeiten");
    addTr("menu.view",          "视图",     "View",           "表示",             "Ansicht");
    addTr("menu.help",          "帮助",     "Help",           "ヘルプ",           "Hilfe");
    addTr("menu.new_project",   "新建项目", "New Project",    "新規プロジェクト", "Neues Projekt");
    addTr("menu.open_project",  "打开项目", "Open Project",   "プロジェクトを開く", "Projekt öffnen");
    addTr("menu.save_project",  "保存项目", "Save Project",   "プロジェクトを保存", "Projekt speichern");
    addTr("menu.close_project", "关闭项目", "Close Project",  "プロジェクトを閉じる", "Projekt schließen");
    addTr("menu.exit",          "退出",     "Exit",           "終了",             "Beenden");
    addTr("menu.undo",          "撤销",     "Undo",           "元に戻す",         "Rückgängig");
    addTr("menu.redo",          "重做",     "Redo",           "やり直す",         "Wiederholen");
    addTr("menu.fullscreen",    "全屏",     "Fullscreen",     "フルスクリーン",   "Vollbild");
    addTr("menu.theme",         "切换主题", "Toggle Theme",   "テーマ切替",       "Design wechseln");
    addTr("menu.shortcuts",     "快捷键",   "Shortcuts",      "ショートカット",   "Tastenkürzel");
    addTr("menu.about",         "关于",     "About",          "バージョン情報",   "Über");
    addTr("menu.settings",      "设置",     "Settings",       "設定",             "Einstellungen");
    addTr("menu.language",      "语言",     "Language",       "言語",             "Sprache");

    // ===== 项目页 =====
    addTr("project.welcome",       "欢迎使用 OmniMatch",              "Welcome to OmniMatch",
           "OmniMatch へようこそ",                                      "Willkommen bei OmniMatch");
    addTr("project.subtitle",      "纯 C++ 全流程深度学习视觉平台",    "Pure C++ Deep Learning Vision Platform",
           "純粋な C++ ディープラーニングビジョンプラットフォーム",       "Reine C++ Deep-Learning-Visions-Plattform");
    addTr("project.new",           "新建项目",  "New Project",    "新規プロジェクト",   "Neues Projekt");
    addTr("project.open",          "打开项目",  "Open Project",   "プロジェクトを開く", "Projekt öffnen");
    addTr("project.open_folder",   "打开文件夹", "Open Folder",   "フォルダーを開く",   "Ordner öffnen");
    addTr("project.recent",        "最近项目",  "Recent Projects", "最近のプロジェクト", "Letzte Projekte");
    addTr("project.name",          "项目名称",  "Project Name",   "プロジェクト名",     "Projektname");
    addTr("project.task_type",     "任务类型",  "Task Type",      "タスクタイプ",       "Aufgabentyp");
    addTr("project.path",          "项目路径",  "Project Path",   "プロジェクトパス",   "Projektpfad");
    addTr("project.created",       "创建时间",  "Created",        "作成日時",           "Erstellt");
    addTr("project.images",        "图像数量",  "Images",         "画像数",             "Bilder");
    addTr("project.labeled",       "已标注",    "Labeled",        "ラベル済み",         "Beschriftet");

    // ===== 图库页 =====
    addTr("gallery.import_images",   "导入图像",   "Import Images",   "画像をインポート",   "Bilder importieren");
    addTr("gallery.import_folder",   "导入文件夹", "Import Folder",   "フォルダーをインポート", "Ordner importieren");
    addTr("gallery.delete_selected", "删除选中",   "Delete Selected", "選択を削除",         "Auswahl löschen");
    addTr("gallery.search",          "搜索...",    "Search...",       "検索...",             "Suchen...");
    addTr("gallery.total",           "总计",       "Total",           "合計",               "Gesamt");
    addTr("gallery.labeled",         "已标注",     "Labeled",         "ラベル済み",         "Beschriftet");
    addTr("gallery.unlabeled",       "未标注",     "Unlabeled",       "未ラベル",           "Unbeschriftet");
    addTr("gallery.sort_name",       "按名称排序", "Sort by Name",    "名前順",             "Nach Name sortieren");
    addTr("gallery.sort_date",       "按日期排序", "Sort by Date",    "日付順",             "Nach Datum sortieren");
    addTr("gallery.filter_all",      "全部显示",   "Show All",        "すべて表示",         "Alle anzeigen");

    // ===== 图像页 / 标注工具 =====
    addTr("image.select_tool",   "选择工具",   "Select Tool",    "選択ツール",       "Auswahlwerkzeug");
    addTr("image.rect_tool",     "矩形工具",   "Rectangle Tool", "矩形ツール",       "Rechteckwerkzeug");
    addTr("image.polygon_tool",  "多边形工具", "Polygon Tool",   "ポリゴンツール",   "Polygonwerkzeug");
    addTr("image.brush_tool",    "画笔工具",   "Brush Tool",     "ブラシツール",     "Pinselwerkzeug");
    addTr("image.brush_size",    "笔刷大小",   "Brush Size",     "ブラシサイズ",     "Pinselgröße");
    addTr("image.assign_label",  "分配标签",   "Assign Label",   "ラベルを割り当て", "Label zuweisen");
    addTr("image.manage_labels", "管理标签",   "Manage Labels",  "ラベル管理",       "Labels verwalten");
    addTr("image.annotations",   "标注列表",   "Annotations",    "アノテーション一覧", "Annotationsliste");
    addTr("image.delete_annot",  "删除标注",   "Delete Annotation", "アノテーション削除", "Annotation löschen");
    addTr("image.show_annots",   "显示标注",   "Show Annotations",  "アノテーション表示", "Annotationen anzeigen");
    addTr("image.prev",          "上一张",     "Previous",       "前へ",             "Vorheriges");
    addTr("image.next",          "下一张",     "Next",           "次へ",             "Nächstes");
    addTr("image.fit_view",      "适应视图",   "Fit View",       "画面に合わせる",   "Ansicht anpassen");
    addTr("image.actual_size",   "实际大小",   "Actual Size",    "実際のサイズ",     "Originalgröße");
    addTr("image.zoom",          "缩放",       "Zoom",           "ズーム",           "Zoom");
    addTr("image.copy_annot",    "复制标注",   "Copy Annotation",   "アノテーションをコピー", "Annotation kopieren");
    addTr("image.paste_annot",   "粘贴标注",   "Paste Annotation",  "アノテーションを貼り付け", "Annotation einfügen");

    // ===== 检查页 =====
    addTr("inspection.run",         "运行检查",   "Run Inspection",  "検査を実行",       "Inspektion starten");
    addTr("inspection.threshold",   "阈值",       "Threshold",       "しきい値",         "Schwellenwert");
    addTr("inspection.result",      "检查结果",   "Result",          "検査結果",         "Ergebnis");
    addTr("inspection.pass",        "合格",       "Pass",            "合格",             "Bestanden");
    addTr("inspection.fail",        "不合格",     "Fail",            "不合格",           "Nicht bestanden");
    addTr("inspection.assign_all",  "批量分配",   "Assign All",      "一括割り当て",     "Alle zuweisen");

    // ===== 拆分页 =====
    addTr("split.random",        "随机拆分",   "Random Split",    "ランダム分割",     "Zufällige Aufteilung");
    addTr("split.stratified",    "分层采样",   "Stratified Split","層化分割",         "Geschichtete Aufteilung");
    addTr("split.train_ratio",   "训练集比例", "Train Ratio",     "訓練セット比率",   "Trainingsanteil");
    addTr("split.val_ratio",     "验证集比例", "Validation Ratio","検証セット比率",   "Validierungsanteil");
    addTr("split.test_ratio",    "测试集比例", "Test Ratio",      "テストセット比率", "Testanteil");
    addTr("split.execute",       "执行拆分",   "Execute Split",   "分割を実行",       "Aufteilung ausführen");
    addTr("split.statistics",    "统计分布",   "Statistics",      "統計分布",         "Statistiken");

    // ===== 训练页 =====
    addTr("train.start",           "开始训练",   "Start Training",   "トレーニング開始",   "Training starten");
    addTr("train.stop",            "停止",       "Stop",             "停止",               "Stopp");
    addTr("train.pause",           "暂停",       "Pause",            "一時停止",           "Pause");
    addTr("train.resume",          "继续",       "Resume",           "再開",               "Fortsetzen");
    addTr("train.model_config",    "模型配置",   "Model Configuration", "モデル設定",      "Modellkonfiguration");
    addTr("train.hyperparams",     "超参数",     "Hyperparameters",  "ハイパーパラメータ", "Hyperparameter");
    addTr("train.augmentation",    "数据增强",   "Data Augmentation", "データ拡張",        "Datenaugmentation");
    addTr("train.framework",       "框架",       "Framework",        "フレームワーク",     "Framework");
    addTr("train.architecture",    "架构",       "Architecture",     "アーキテクチャ",     "Architektur");
    addTr("train.device",          "设备",       "Device",           "デバイス",           "Gerät");
    addTr("train.optimizer",       "优化器",     "Optimizer",        "オプティマイザ",     "Optimierer");
    addTr("train.scheduler",       "调度器",     "Scheduler",        "スケジューラ",       "Planer");
    addTr("train.learning_rate",   "学习率",     "Learning Rate",    "学習率",             "Lernrate");
    addTr("train.batch_size",      "批量大小",   "Batch Size",       "バッチサイズ",       "Batchgröße");
    addTr("train.epochs",          "训练轮次",   "Epochs",           "エポック数",         "Epochen");
    addTr("train.input_size",      "输入尺寸",   "Input Size",       "入力サイズ",         "Eingabegröße");
    addTr("train.patience",        "早停耐心",   "Early Stopping Patience", "早期停止忍耐値", "Frühstoppgeduld");
    addTr("train.progress",        "训练进度",   "Training Progress", "トレーニング進捗",  "Trainingsfortschritt");
    addTr("train.loss",            "损失值",     "Loss",             "損失値",             "Verlust");
    addTr("train.epoch_label",     "第 %1 轮",   "Epoch %1",         "エポック %1",        "Epoche %1");
    addTr("train.elapsed",         "已用时间",   "Elapsed Time",     "経過時間",           "Verstrichene Zeit");
    addTr("train.remaining",       "剩余时间",   "Remaining Time",   "残り時間",           "Verbleibende Zeit");

    // ===== 评估页 =====
    addTr("eval.run",              "运行评估",   "Run Evaluation",    "評価を実行",        "Bewertung starten");
    addTr("eval.clear",            "清除结果",   "Clear Results",     "結果をクリア",      "Ergebnisse löschen");
    addTr("eval.export_csv",       "导出 CSV",   "Export CSV",        "CSV エクスポート",  "CSV exportieren");
    addTr("eval.export_html",      "导出报告",   "Export Report",     "レポートをエクスポート", "Bericht exportieren");
    addTr("eval.accuracy",         "准确率",     "Accuracy",          "精度",              "Genauigkeit");
    addTr("eval.precision",        "精确率",     "Precision",         "適合率",            "Präzision");
    addTr("eval.recall",           "召回率",     "Recall",            "再現率",            "Trefferquote");
    addTr("eval.f1_score",         "F1 分数",    "F1 Score",          "F1 スコア",         "F1-Wert");
    addTr("eval.confusion_matrix", "混淆矩阵",   "Confusion Matrix",  "混同行列",          "Konfusionsmatrix");
    addTr("eval.iou",              "IoU",        "IoU",               "IoU",               "IoU");
    addTr("eval.miou",             "平均 IoU",   "Mean IoU",          "平均 IoU",          "Mittlerer IoU");
    addTr("eval.map",              "mAP",        "mAP",               "mAP",               "mAP");

    // ===== 导出页 =====
    addTr("export.start",          "开始导出",   "Start Export",      "エクスポート開始",  "Export starten");
    addTr("export.format",         "导出格式",   "Export Format",     "エクスポート形式",  "Exportformat");
    addTr("export.precision",      "精度",       "Precision",         "精度",              "Genauigkeit");
    addTr("export.output_dir",     "输出目录",   "Output Directory",  "出力ディレクトリ",  "Ausgabeverzeichnis");
    addTr("export.onnx",           "ONNX 格式",  "ONNX Format",      "ONNX 形式",         "ONNX-Format");
    addTr("export.tensorrt",       "TensorRT 格式", "TensorRT Format", "TensorRT 形式",    "TensorRT-Format");
    addTr("export.openvino",       "OpenVINO 格式",  "OpenVINO Format", "OpenVINO 形式",   "OpenVINO-Format");
    addTr("export.success",        "导出成功",   "Export Successful", "エクスポート成功",  "Export erfolgreich");
    addTr("export.failed",         "导出失败",   "Export Failed",     "エクスポート失敗",  "Export fehlgeschlagen");

    // ===== 通用按钮 =====
    addTr("common.ok",            "确定",   "OK",          "OK",         "OK");
    addTr("common.cancel",        "取消",   "Cancel",      "キャンセル", "Abbrechen");
    addTr("common.apply",         "应用",   "Apply",       "適用",       "Anwenden");
    addTr("common.close",         "关闭",   "Close",       "閉じる",     "Schließen");
    addTr("common.browse",        "浏览",   "Browse",      "参照",       "Durchsuchen");
    addTr("common.save",          "保存",   "Save",        "保存",       "Speichern");
    addTr("common.delete",        "删除",   "Delete",      "削除",       "Löschen");
    addTr("common.reset",         "重置",   "Reset",       "リセット",   "Zurücksetzen");
    addTr("common.refresh",       "刷新",   "Refresh",     "更新",       "Aktualisieren");
    addTr("common.all",           "全部",   "All",         "すべて",     "Alle");
    addTr("common.none",          "无",     "None",        "なし",       "Keine");
    addTr("common.train",         "训练",   "Train",       "訓練",       "Training");
    addTr("common.validation",    "验证",   "Validation",  "検証",       "Validierung");
    addTr("common.test",          "测试",   "Test",        "テスト",     "Test");
    addTr("common.unassigned",    "未分配", "Unassigned",  "未割り当て", "Nicht zugewiesen");
    addTr("common.yes",           "是",     "Yes",         "はい",       "Ja");
    addTr("common.no",            "否",     "No",          "いいえ",     "Nein");
    addTr("common.error",         "错误",   "Error",       "エラー",     "Fehler");
    addTr("common.warning",       "警告",   "Warning",     "警告",       "Warnung");
    addTr("common.info",          "提示",   "Information", "情報",       "Information");
    addTr("common.confirm",       "确认",   "Confirm",     "確認",       "Bestätigen");
    addTr("common.loading",       "加载中...", "Loading...",  "読み込み中...", "Laden...");
    addTr("common.ready",         "就绪",   "Ready",       "準備完了",   "Bereit");
    addTr("common.copy",          "复制",   "Copy",        "コピー",     "Kopieren");
    addTr("common.paste",         "粘贴",   "Paste",       "貼り付け",   "Einfügen");
    addTr("common.select_all",    "全选",   "Select All",  "すべて選択", "Alles auswählen");

    // ===== 任务类型 =====
    addTr("task.classification",         "图像分类",       "Classification",          "画像分類",           "Klassifizierung");
    addTr("task.object_detection",       "目标检测",       "Object Detection",        "物体検出",           "Objekterkennung");
    addTr("task.semantic_segmentation",  "语义分割",       "Semantic Segmentation",   "セマンティックセグメンテーション", "Semantische Segmentierung");
    addTr("task.anomaly_detection",      "异常检测",       "Anomaly Detection",       "異常検出",           "Anomalieerkennung");
    addTr("task.ocr",                    "OCR 文字识别",   "OCR",                     "OCR 文字認識",       "OCR-Texterkennung");
    addTr("task.instance_segmentation",  "实例分割",       "Instance Segmentation",   "インスタンスセグメンテーション", "Instanzsegmentierung");
    addTr("task.zeroshot_detection",     "零样本检测",     "Zero-Shot Detection",     "ゼロショット検出",   "Zero-Shot-Erkennung");
    addTr("task.zeroshot_anomaly",       "零样本异常检测", "Zero-Shot Anomaly Detection", "ゼロショット異常検出", "Zero-Shot-Anomalieerkennung");

    // ===== 状态栏消息 =====
    addTr("status.no_project",     "未打开项目",         "No Project Open",          "プロジェクト未開",       "Kein Projekt geöffnet");
    addTr("status.project_loaded", "项目已加载: %1",     "Project Loaded: %1",       "プロジェクトを読み込みました: %1", "Projekt geladen: %1");
    addTr("status.training",       "训练中...",           "Training...",              "トレーニング中...",      "Training läuft...");
    addTr("status.evaluating",     "评估中...",           "Evaluating...",            "評価中...",              "Bewertung läuft...");
    addTr("status.exporting",      "导出中...",           "Exporting...",             "エクスポート中...",      "Export läuft...");
    addTr("status.saved",          "已保存",             "Saved",                    "保存しました",           "Gespeichert");
    addTr("status.gpu_available",  "GPU 可用",           "GPU Available",            "GPU 利用可能",           "GPU verfügbar");
    addTr("status.cpu_only",       "仅 CPU",             "CPU Only",                 "CPU のみ",               "Nur CPU");

    // ===== 对话框标题 =====
    addTr("dialog.new_project",    "新建项目",   "New Project",    "新規プロジェクト",   "Neues Projekt");
    addTr("dialog.label_mgmt",     "标签管理",   "Label Management", "ラベル管理",       "Label-Verwaltung");
    addTr("dialog.import_images",  "导入图像",   "Import Images",  "画像インポート",     "Bilder importieren");
    addTr("dialog.export_model",   "导出模型",   "Export Model",   "モデルエクスポート", "Modell exportieren");
    addTr("dialog.confirm_delete", "确认删除",   "Confirm Delete", "削除の確認",         "Löschen bestätigen");
    addTr("dialog.unsaved_changes","未保存的更改", "Unsaved Changes", "未保存の変更",    "Nicht gespeicherte Änderungen");
    addTr("dialog.update_available", "更新可用", "Update Available", "アップデート利用可能", "Update verfügbar");

    // ===== 缺陷严重度 =====
    addTr("severity.none",    "未指定", "Unspecified", "未指定",   "Nicht angegeben");
    addTr("severity.low",     "低",     "Low",         "低",       "Niedrig");
    addTr("severity.medium",  "中",     "Medium",      "中",       "Mittel");
    addTr("severity.high",    "高",     "High",        "高",       "Hoch");
    addTr("severity.ignore",  "忽略",   "Ignore",      "無視",     "Ignorieren");
}
