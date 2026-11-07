# النشر: الروابط الجاهزة وطريقة التحديث

| الشيء | الرابط |
|---|---|
| الكود والتقرير | https://github.com/tgfdru/saudi-legal-rag |
| الواجهة الحية | https://huggingface.co/spaces/tgfdru/saudi-legal-rag |
| تشغيل الـ notebook | زر **Open in Colab** في README |

## كيف يشتغل الـ Space؟
الـ Space فيه ثلاث ملفات بس، موجودة نسخة منها في `deploy/space_*`. ملف `app.py` يسحب المشروع من GitHub عند كل بناء ويشغّل `app/gradio_app.py`.
**يعني أي تعديل ترفعه على GitHub يوصل للواجهة تلقائياً** بعد ما تسوي **Restart** للـ Space، من Settings ثم Factory rebuild.

## قبل العرض
- **افتح رابط الـ Space قبل العرض بربع ساعة.** الـ Space المجاني ينام بعد ٤٨ ساعة بدون استخدام، ويحتاج دقائق عشان يصحى.
- **الاحتياط:** في آخر الـ notebook على Colab، حط `LAUNCH_DEMO = True` وشغّل الخلية. بيطلع لك رابط `gradio.live` مؤقت.
- **على الـ CPU المجاني الـ reranker مقفل،** والواجهة تشتغل بـ bge-m3 لحاله. نتيجته R@5 = 0.91.
