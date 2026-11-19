package neuralforge.data;

import neuralforge.TestData;
import java.nio.charset.StandardCharsets;
import static neuralforge.TestHelper.*;

public class CsvDetectTest{

    public static void main(String[] args){
        testDelimiters();
        testDelimiterEdgeCases();
        testReadAuto();
        testErrorMessages();
        testBinaryFile();
        finish();
    }

    private static String path(String content){
        return TestData.writeText(".csv", content);
    }

    private static String messageOf(Runnable code){
        try{
            code.run();
        }catch(DataFormatException e){
            return e.getMessage();
        }catch(RuntimeException e){
            return "wrong exception type " + e.getClass().getName();
        }
        return null;
    }

    private static void testDelimiters(){
        check("comma", CsvReader.detectDelimiter(path("a,b,c\n1,2,3\n4,5,6\n")) == ',');
        check("semicolon", CsvReader.detectDelimiter(path("a;b;c\n1;2;3\n4;5;6\n")) == ';');
        check("tab", CsvReader.detectDelimiter(path("a\tb\tc\n1\t2\t3\n")) == '\t');
        check("bar", CsvReader.detectDelimiter(path("a|b|c\n1|2|3\n")) == '|');
    }

    private static void testDelimiterEdgeCases(){

        check("quoted commas", CsvReader.detectDelimiter(path("name;note\n\"a,b,c\";1\n\"d,e\";2\n")) == ';');

        check("decimal commas", CsvReader.detectDelimiter(path("a;b\n1,5;2,5\n3,5;4,5\n")) == ';');
        check("one column falls back to comma", CsvReader.detectDelimiter(path("a\n1\n2\n")) == ',');
        check("inconsistent counts fall back to comma", CsvReader.detectDelimiter(path("a;b;c\n1;2\n3;4;5;6\n")) == ',');
        check("empty lines are skipped", CsvReader.detectDelimiter(path("\n\na;b\n\n1;2\n")) == ';');
        check("byte order mark is skipped", CsvReader.detectDelimiter(path("\uFEFFa;b\n1;2\n")) == ';');
        check("most frequent consistent delimiter wins", CsvReader.detectDelimiter(path("a,b;c;d\n1,2;3;4\n")) == ';');
    }

    private static void testReadAuto(){
        CsvTable table = CsvReader.readAuto(path("name;score;grade\nann;3,5;a\nbob;4,5;b\n"), true);
        check("three columns", table.columnCount() == 3 && table.rowCount() == 2);
        check("cells", table.get(0, 1).equals("3,5") && table.get(1, 2).equals("b"));
        check("comma files still work", CsvReader.readAuto(path("a,b\n1,2\n"), true).columnCount() == 2);
    }

    private static void testErrorMessages(){
        String missing = messageOf(() -> CsvReader.read("no/such/place/data.csv", true));
        check("file not found, got " + missing, missing != null && missing.contains("file not found") && missing.contains("data.csv"));

        String folder = messageOf(() -> CsvReader.read(System.getProperty("java.io.tmpdir"), true));
        check("a folder, got " + folder, folder != null && folder.contains("is a folder"));

        String ragged = messageOf(() -> CsvReader.read(path("a,b,c\n1,2,3\n4,5\n"), true));
        check("ragged row, got " + ragged, ragged != null && ragged.contains("line 3 has 2 fields but the first line has 3") && ragged.contains("delimiter"));

        String quote = messageOf(() -> CsvReader.read(path("a,b\n\"x,1\n"), true));
        check("open quote names the line, got " + quote, quote != null && quote.contains("line 2") && quote.contains("never closed"));

        String empty = messageOf(() -> CsvReader.read(path(""), true));
        check("empty file, got " + empty, empty != null && empty.contains("is empty"));

        check("the exception is an IllegalArgumentException too", throwsIllegalArgument(() -> CsvReader.read("no/such/file.csv", true)));
    }

    private static boolean throwsIllegalArgument(Runnable code){
        try{
            code.run();
        }catch(IllegalArgumentException e){
            return true;
        }catch(RuntimeException e){
            return false;
        }
        return false;
    }

    private static void testBinaryFile(){
        byte[] bytes = new byte[200];
        for(int i = 0; i < bytes.length; i++){
            bytes[i] = (byte) (0x80 + (i % 64));
        }
        String file = TestData.writeBytes(".csv", bytes);
        String message = messageOf(() -> CsvReader.read(file, true));
        check("binary file, got " + message, message != null && message.contains("not a plain text file") && message.contains("Excel"));
        check("delimiter detection gives the same message", String.valueOf(messageOf(() -> CsvReader.detectDelimiter(file))).contains("not a plain text file"));

        check("plain text with accents is fine", CsvReader.read(TestData.writeBytes(".csv", "name,city\nJosé,Zürich\n".getBytes(StandardCharsets.UTF_8)), true)
            .get(0, 1).equals("Zürich"));
    }
}
