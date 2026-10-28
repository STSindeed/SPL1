package neuralforge;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPOutputStream;

public class TestData{

    public static String writeText(String suffix, String content){
        return writeBytes(suffix, content.getBytes(StandardCharsets.UTF_8));
    }

    public static String writeBytes(String suffix, byte[] content){
        try{
            Path file = Files.createTempFile("neuralforge", suffix);
            file.toFile().deleteOnExit();
            Files.write(file, content);
            return file.toString();
        }catch(IOException e){
            throw new UncheckedIOException(e);
        }
    }

    public static byte[] idxBytes(int type, int[] dims, byte[] data){
        try{
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            DataOutputStream out = new DataOutputStream(bytes);
            out.writeByte(0);
            out.writeByte(0);
            out.writeByte(type);
            out.writeByte(dims.length);
            for(int d : dims){
                out.writeInt(d);
            }
            out.write(data);
            out.flush();
            return bytes.toByteArray();
        }catch(IOException e){
            throw new UncheckedIOException(e);
        }
    }

    public static byte[] idxBytes(int[] dims, byte[] data){
        return idxBytes(0x08, dims, data);
    }

    public static byte[] gzip(byte[] raw){
        try{
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            GZIPOutputStream out = new GZIPOutputStream(bytes);
            out.write(raw);
            out.close();
            return bytes.toByteArray();
        }catch(IOException e){
            throw new UncheckedIOException(e);
        }
    }

    public static String messageOf(Runnable code){
        try{
            code.run();
            return null;
        }catch(RuntimeException e){
            return e.getMessage();
        }
    }

    public static byte[] bytes(int... values){
        byte[] result = new byte[values.length];
        for(int i = 0; i < values.length; i++){
            result[i] = (byte) values[i];
        }
        return result;
    }
}
