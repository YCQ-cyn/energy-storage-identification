import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.awt.Desktop;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

public final class StorageScreeningApplication {
    private static final long MAX_UPLOAD_BYTES = 512L * 1024L * 1024L;
    private static final Duration PROCESS_TIMEOUT = Duration.ofMinutes(60);
    private static final Duration JOB_RETENTION = Duration.ofHours(24);

    private final Path projectRoot;
    private final Path staticRoot;
    private final Path workRoot;
    private final String pythonExecutable;
    private final int port;
    private final Semaphore inferenceSlot = new Semaphore(1);
    private final Map<String, Path> downloadFiles = new ConcurrentHashMap<>();

    private StorageScreeningApplication(
            Path projectRoot, String pythonExecutable, int port) {
        this.projectRoot = projectRoot.toAbsolutePath().normalize();
        this.staticRoot = this.projectRoot.resolve("web/static");
        this.workRoot = this.projectRoot.resolve("web/work");
        this.pythonExecutable = pythonExecutable;
        this.port = port;
    }

    public static void main(String[] arguments) throws Exception {
        Map<String, String> options = parseArguments(arguments);
        Path root = Path.of(options.getOrDefault("root", "."));
        String python = options.getOrDefault("python", "python");
        int port = Integer.parseInt(options.getOrDefault("port", "8765"));
        new StorageScreeningApplication(root, python, port).start();
    }

    private void start() throws Exception {
        validateRuntime();
        Files.createDirectories(workRoot);
        cleanupExpiredJobs();

        HttpServer server = HttpServer.create(
                new InetSocketAddress("127.0.0.1", port), 0);
        server.createContext("/api/analyze", this::handleAnalyze);
        server.createContext("/api/download", this::handleDownload);
        server.createContext("/api/health", this::handleHealth);
        server.createContext("/", this::handleStatic);
        server.setExecutor(Executors.newFixedThreadPool(
                Math.max(4, Runtime.getRuntime().availableProcessors())));
        server.start();

        String address = "http://127.0.0.1:" + port;
        System.out.println("[Web] 用户侧储能智能辅助排查服务已启动");
        System.out.println("[Web] " + address);
        openBrowser(address);
    }

    private void validateRuntime() throws IOException {
        if (!Files.isRegularFile(projectRoot.resolve("infer.py"))) {
            throw new IOException("未找到 infer.py：" + projectRoot);
        }
        if (!Files.isRegularFile(
                projectRoot.resolve("train_student/best_model_student.pt"))) {
            throw new IOException("未找到学生模型 train_student/best_model_student.pt");
        }
        if (!Files.isDirectory(staticRoot)) {
            throw new IOException("未找到前端资源目录：" + staticRoot);
        }
    }

    private void handleAnalyze(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendJson(exchange, 405, errorJson("仅支持POST上传"));
            return;
        }
        Map<String, String> query = parseQuery(exchange.getRequestURI());
        String policy = query.getOrDefault("policy", "new");
        if (!policy.equals("new") && !policy.equals("old")) {
            sendJson(exchange, 400, errorJson("电价政策仅支持new或old"));
            return;
        }
        String originalName = query.getOrDefault("filename", "input.xlsx");
        String extension = fileExtension(originalName);
        if (!List.of("xlsx", "csv").contains(extension)) {
            sendJson(exchange, 400, errorJson("仅支持xlsx或csv文件；xls请先另存为xlsx"));
            return;
        }
        long declaredLength = parseLong(
                exchange.getRequestHeaders().getFirst("Content-Length"), -1L);
        if (declaredLength > MAX_UPLOAD_BYTES) {
            sendJson(exchange, 413, errorJson("上传文件不能超过512MB"));
            return;
        }

        String jobId = UUID.randomUUID().toString().replace("-", "");
        Path jobRoot = workRoot.resolve(jobId);
        Path inputPath = jobRoot.resolve("input." + extension);
        Path resultPath = jobRoot.resolve("用户侧储能排查结果.xlsx");
        Path previewPath = jobRoot.resolve("preview.json");
        Path logPath = jobRoot.resolve("infer.log");
        Files.createDirectories(jobRoot);

        try {
            copyUpload(exchange.getRequestBody(), inputPath);
        } catch (UploadTooLargeException exception) {
            deleteDirectory(jobRoot);
            sendJson(exchange, 413, errorJson(exception.getMessage()));
            return;
        } catch (IOException exception) {
            deleteDirectory(jobRoot);
            sendJson(exchange, 400, errorJson("文件保存失败：" + exception.getMessage()));
            return;
        }

        boolean acquired = false;
        try {
            inferenceSlot.acquire();
            acquired = true;
            int exitCode = runInference(
                    inputPath, resultPath, previewPath, logPath, policy);
            if (exitCode != 0) {
                sendJson(exchange, 422, errorJson(readLogTail(logPath)));
                return;
            }
            if (!Files.isRegularFile(resultPath) || !Files.isRegularFile(previewPath)) {
                sendJson(exchange, 500, errorJson("推理完成但结果文件未生成"));
                return;
            }
            downloadFiles.put(jobId, resultPath);
            String rows = Files.readString(previewPath, StandardCharsets.UTF_8);
            String response = "{" +
                    "\"jobId\":\"" + escapeJson(jobId) + "\"," +
                    "\"fileName\":\"用户侧储能排查结果.xlsx\"," +
                    "\"downloadUrl\":\"/api/download?id=" + jobId + "\"," +
                    "\"rows\":" + rows +
                    "}";
            sendJson(exchange, 200, response);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            sendJson(exchange, 503, errorJson("任务被中断，请重新提交"));
        } catch (Exception exception) {
            sendJson(exchange, 500, errorJson("服务处理失败：" + exception.getMessage()));
        } finally {
            if (acquired) {
                inferenceSlot.release();
            }
        }
    }

    private int runInference(
            Path inputPath,
            Path resultPath,
            Path previewPath,
            Path logPath,
            String policy) throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder(
                pythonExecutable,
                projectRoot.resolve("infer.py").toString(),
                "--input_path", inputPath.toString(),
                "--output_path", resultPath.toString(),
                "--policy_name", policy,
                "--compact_output",
                "--preview_json", previewPath.toString());
        builder.directory(projectRoot.toFile());
        builder.redirectErrorStream(true);
        builder.redirectOutput(logPath.toFile());
        builder.environment().put("PYTHONIOENCODING", "utf-8");
        builder.environment().put("PYTHONUTF8", "1");
        Process process = builder.start();
        boolean finished = process.waitFor(PROCESS_TIMEOUT.toMinutes(), TimeUnit.MINUTES);
        if (!finished) {
            process.destroyForcibly();
            Files.writeString(
                    logPath,
                    System.lineSeparator() + "推理超时，已终止任务",
                    StandardCharsets.UTF_8,
                    java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.APPEND);
            return 124;
        }
        return process.exitValue();
    }

    private void handleDownload(HttpExchange exchange) throws IOException {
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendJson(exchange, 405, errorJson("仅支持GET下载"));
            return;
        }
        String jobId = parseQuery(exchange.getRequestURI()).get("id");
        Path file = jobId == null ? null : downloadFiles.get(jobId);
        if (file == null || !Files.isRegularFile(file)) {
            sendJson(exchange, 404, errorJson("结果不存在或服务已重启，请重新分析"));
            return;
        }
        Headers headers = exchange.getResponseHeaders();
        headers.set(
                "Content-Type",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
        String encodedName = URLEncoder.encode(
                "用户侧储能排查结果.xlsx", StandardCharsets.UTF_8).replace("+", "%20");
        headers.set("Content-Disposition", "attachment; filename*=UTF-8''" + encodedName);
        headers.set("Cache-Control", "no-store");
        exchange.sendResponseHeaders(200, Files.size(file));
        try (OutputStream output = exchange.getResponseBody()) {
            Files.copy(file, output);
        }
    }

    private void handleHealth(HttpExchange exchange) throws IOException {
        sendJson(exchange, 200, "{\"status\":\"ok\"}");
    }

    private void handleStatic(HttpExchange exchange) throws IOException {
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendText(exchange, 405, "text/plain; charset=utf-8", "Method Not Allowed");
            return;
        }
        String path = exchange.getRequestURI().getPath();
        String fileName = switch (path) {
            case "/", "/index.html" -> "index.html";
            case "/styles.css" -> "styles.css";
            case "/app.js" -> "app.js";
            default -> null;
        };
        if (fileName == null) {
            sendText(exchange, 404, "text/plain; charset=utf-8", "Not Found");
            return;
        }
        Path file = staticRoot.resolve(fileName).normalize();
        if (!file.startsWith(staticRoot) || !Files.isRegularFile(file)) {
            sendText(exchange, 404, "text/plain; charset=utf-8", "Not Found");
            return;
        }
        String contentType = switch (fileName) {
            case "styles.css" -> "text/css; charset=utf-8";
            case "app.js" -> "text/javascript; charset=utf-8";
            default -> "text/html; charset=utf-8";
        };
        byte[] body = Files.readAllBytes(file);
        exchange.getResponseHeaders().set("Cache-Control", "no-cache");
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(200, body.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(body);
        }
    }

    private static void copyUpload(InputStream input, Path destination)
            throws IOException, UploadTooLargeException {
        Path temporary = destination.resolveSibling(destination.getFileName() + ".part");
        long total = 0L;
        byte[] buffer = new byte[1024 * 1024];
        try (OutputStream output = Files.newOutputStream(temporary)) {
            int read;
            while ((read = input.read(buffer)) >= 0) {
                total += read;
                if (total > MAX_UPLOAD_BYTES) {
                    throw new UploadTooLargeException("上传文件不能超过512MB");
                }
                output.write(buffer, 0, read);
            }
        } catch (Exception exception) {
            Files.deleteIfExists(temporary);
            throw exception;
        }
        Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING);
    }

    private void cleanupExpiredJobs() throws IOException {
        if (!Files.isDirectory(workRoot)) {
            return;
        }
        Instant cutoff = Instant.now().minus(JOB_RETENTION);
        try (var entries = Files.list(workRoot)) {
            entries.filter(Files::isDirectory).forEach(path -> {
                try {
                    if (Files.getLastModifiedTime(path).toInstant().isBefore(cutoff)) {
                        deleteDirectory(path);
                    }
                } catch (IOException ignored) {
                }
            });
        }
    }

    private static void deleteDirectory(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }
        try (var paths = Files.walk(root)) {
            paths.sorted((left, right) -> right.compareTo(left)).forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException ignored) {
                }
            });
        }
    }

    private static Map<String, String> parseArguments(String[] arguments) {
        Map<String, String> values = new HashMap<>();
        for (int index = 0; index + 1 < arguments.length; index += 2) {
            String key = arguments[index].replaceFirst("^--", "");
            values.put(key, arguments[index + 1]);
        }
        return values;
    }

    private static Map<String, String> parseQuery(URI uri) {
        Map<String, String> values = new HashMap<>();
        String query = uri.getRawQuery();
        if (query == null || query.isBlank()) {
            return values;
        }
        for (String pair : query.split("&")) {
            String[] parts = pair.split("=", 2);
            String key = URLDecoder.decode(parts[0], StandardCharsets.UTF_8);
            String value = parts.length == 2
                    ? URLDecoder.decode(parts[1], StandardCharsets.UTF_8)
                    : "";
            values.put(key, value);
        }
        return values;
    }

    private static String fileExtension(String fileName) {
        int index = fileName.lastIndexOf('.');
        if (index < 0 || index == fileName.length() - 1) {
            return "";
        }
        return fileName.substring(index + 1).toLowerCase(Locale.ROOT);
    }

    private static long parseLong(String value, long fallback) {
        try {
            return Long.parseLong(value);
        } catch (Exception exception) {
            return fallback;
        }
    }

    private static String readLogTail(Path logPath) {
        try {
            String log = Files.readString(logPath, StandardCharsets.UTF_8).trim();
            if (log.isEmpty()) {
                return "推理失败，未返回错误信息";
            }
            int limit = 3000;
            return log.length() <= limit ? log : log.substring(log.length() - limit);
        } catch (IOException exception) {
            return "推理失败，且无法读取错误日志";
        }
    }

    private static String errorJson(String message) {
        return "{\"error\":\"" + escapeJson(message) + "\"}";
    }

    private static String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        StringBuilder escaped = new StringBuilder(value.length() + 16);
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            switch (character) {
                case '"' -> escaped.append("\\\"");
                case '\\' -> escaped.append("\\\\");
                case '\n' -> escaped.append("\\n");
                case '\r' -> escaped.append("\\r");
                case '\t' -> escaped.append("\\t");
                default -> {
                    if (character < 0x20) {
                        escaped.append(String.format("\\u%04x", (int) character));
                    } else {
                        escaped.append(character);
                    }
                }
            }
        }
        return escaped.toString();
    }

    private static void sendJson(HttpExchange exchange, int status, String json)
            throws IOException {
        sendText(exchange, status, "application/json; charset=utf-8", json);
    }

    private static void sendText(
            HttpExchange exchange, int status, String contentType, String value)
            throws IOException {
        byte[] body = value.getBytes(StandardCharsets.UTF_8);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", contentType);
        headers.set("Cache-Control", "no-store");
        exchange.sendResponseHeaders(status, body.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(body);
        }
    }

    private static void openBrowser(String address) {
        if (!Desktop.isDesktopSupported()) {
            return;
        }
        try {
            Desktop.getDesktop().browse(URI.create(address));
        } catch (Exception exception) {
            System.out.println("[Web] 请手动打开：" + address);
        }
    }

    private static final class UploadTooLargeException extends IOException {
        private UploadTooLargeException(String message) {
            super(message);
        }
    }
}
